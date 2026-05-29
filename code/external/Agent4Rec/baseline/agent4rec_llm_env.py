from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.avatar import Avatar


@dataclass
class UserTraits:
    activity: float
    diversity: float
    conformity: float


class Agent4RecLLMEnv:
    """
    LLM-avatar whole-session environment wrapper for RL training.

    This env keeps the same high-level interface as the numeric baseline env:
    - reset() -> state
    - step(action) -> next_state, reward, done, info
    """

    def __init__(
        self,
        root_dir: str,
        dataset: str = "ml-1m",
        modeltype: str = "SASRec",
        simulation_name: str = "rl_llm_train",
        llm_model: str = "gpt-4o-mini",
        llm_api_style: str = "responses",
        use_wandb: bool = False,
        max_pages: int = 20,
        slate_size: int = 1,
        seed: int = 11,
        user_limit: Optional[int] = None,
        exit_threshold_scale: float = 1.35,
        repetition_penalty_scale: float = 1.20,
        repeat_aversion_scale: float = 1.50,
        fatigue_scale: float = 1.80,
        terminal_score_max: float = 10.0,
        terminal_from_interview: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset = str(dataset)
        self.modeltype = str(modeltype)
        self.simulation_name = str(simulation_name)
        self.llm_model = str(llm_model)
        self.llm_api_style = str(llm_api_style)
        self.use_wandb = bool(use_wandb)

        self.max_pages = int(max_pages)
        self.slate_size = int(slate_size)
        self.exit_threshold_scale = float(exit_threshold_scale)
        self.repetition_penalty_scale = float(repetition_penalty_scale)
        self.repeat_aversion_scale = float(repeat_aversion_scale)
        self.fatigue_scale = float(fatigue_scale)
        self.terminal_score_max = float(terminal_score_max)
        self.terminal_from_interview = bool(terminal_from_interview)

        self.rng = np.random.default_rng(seed)
        self._load_data(user_limit=user_limit)
        self._ensure_storage_dirs()

        self.action_dim = 3
        self.effect_dim = self.genre_dim
        self.state_dim = self.genre_dim + 3 + 4 + self.genre_dim

        self.response_types = ["watch", "like", "align", "dislike"]
        self.response_dim = len(self.response_types)
        self.response_weights = np.array([0.20, 0.85, 0.10, -0.60], dtype=np.float32)
        self.episode_batch_size = 1
        self.single_response = False

        self.current_user: int = 0
        self.page_idx: int = 0
        self.dissatisfaction: float = 0.0
        self.exit_threshold: float = 1.0
        self.recent_genre: np.ndarray = np.zeros(self.genre_dim, dtype=np.float32)
        self.last_genre_repetition: float = 0.0
        self.recent_satisfaction: float = 0.5
        self.shown_mask: np.ndarray = np.zeros(self.n_items, dtype=bool)
        self.session_likes: int = 0
        self.session_shown: int = 0
        self.session_satisfaction_sum: float = 0.0
        self.avatar: Optional[Avatar] = None
        self.current_observation: Optional[np.ndarray] = None

    def _ensure_storage_dirs(self) -> None:
        log_dir = self.root_dir / "storage" / self.dataset / self.modeltype / self.simulation_name / "running_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

    def _load_data(self, user_limit: Optional[int]) -> None:
        sim_dir = self.root_dir / "datasets" / self.dataset / "simulation"
        cf_dir = self.root_dir / "datasets" / self.dataset / "cf_data"

        movie_df = pd.read_csv(sim_dir / "movie_detail.csv")
        movie_df = movie_df.sort_values("movie_id").reset_index(drop=True)
        self.n_items = int(movie_df["movie_id"].max()) + 1

        all_genres = set()
        for g in movie_df["genres"].fillna(""):
            for token in str(g).split("|"):
                token = token.strip()
                if token:
                    all_genres.add(token)
        self.genre_list = sorted(all_genres)
        self.genre2idx = {g: i for i, g in enumerate(self.genre_list)}
        self.genre_dim = len(self.genre_list)

        item_genre = np.zeros((self.n_items, self.genre_dim), dtype=np.float32)
        item_quality = np.zeros(self.n_items, dtype=np.float32)
        item_title = [""] * self.n_items
        item_summary = [""] * self.n_items

        for _, row in movie_df.iterrows():
            iid = int(row["movie_id"])
            if iid < 0 or iid >= self.n_items:
                continue
            item_quality[iid] = float(row["rating"])
            item_title[iid] = str(row["title"])
            item_summary[iid] = str(row.get("summary", ""))
            for g in str(row["genres"]).split("|"):
                g = g.strip()
                if g in self.genre2idx:
                    item_genre[iid, self.genre2idx[g]] = 1.0

        quality_min = float(item_quality.min())
        quality_max = float(item_quality.max())
        self.item_quality = (item_quality - quality_min) / max(quality_max - quality_min, 1e-6)
        self.item_genre = item_genre
        self.item_title = item_title
        self.item_summary = item_summary

        pop = np.zeros(self.n_items, dtype=np.float32)
        user_hist: Dict[int, List[int]] = {}
        with open(cf_dir / "train.txt", "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                uid = int(tokens[0])
                items = [int(x) for x in tokens[1:] if x.isdigit()]
                user_hist[uid] = items
                for iid in items:
                    if 0 <= iid < self.n_items:
                        pop[iid] += 1
        pop = pop / max(pop.max(), 1.0)
        self.item_popularity = pop
        self.item_novelty = 1.0 - pop

        stat_df = pd.read_csv(sim_dir / "user_statistic.csv")
        if "user_id" in stat_df.columns:
            stat_df = stat_df.set_index("user_id")
        stat_df = stat_df.sort_index()

        persona_df = pd.read_csv(sim_dir / "all_personas_like_modify.csv")
        self.persona_df = persona_df

        max_user_id = int(stat_df.index.max())
        user_pref = np.zeros((max_user_id + 1, self.genre_dim), dtype=np.float32)
        user_traits: Dict[int, UserTraits] = {}

        for uid in stat_df.index.tolist():
            uid = int(uid)
            hist = user_hist.get(uid, [])
            if hist:
                gvec = self.item_genre[np.array(hist)].mean(axis=0)
            else:
                gvec = np.ones(self.genre_dim, dtype=np.float32) / max(self.genre_dim, 1)
            gsum = float(gvec.sum())
            if gsum <= 0:
                gvec = np.ones(self.genre_dim, dtype=np.float32) / max(self.genre_dim, 1)
            else:
                gvec = gvec / gsum
            user_pref[uid] = gvec.astype(np.float32)

            row = stat_df.loc[uid]
            user_traits[uid] = UserTraits(
                activity=float(row["activity"]),
                diversity=float(row["diversity"]),
                conformity=float(row["conformity"]),
            )

        user_ids = sorted(user_traits.keys())
        user_ids = [uid for uid in user_ids if 0 <= uid < len(self.persona_df)]
        if user_limit is not None:
            user_ids = user_ids[: int(user_limit)]
        if not user_ids:
            raise ValueError("No valid user ids found for LLM environment.")

        self.user_ids = np.array(user_ids, dtype=np.int32)
        self.user_pref = user_pref
        self.user_traits = user_traits
        self.user_stat_df = stat_df

    def _build_avatar_args(self) -> Namespace:
        return Namespace(
            dataset=self.dataset,
            modeltype=self.modeltype,
            simulation_name=self.simulation_name,
            use_wandb=self.use_wandb,
            llm_model=self.llm_model,
            llm_api_style=self.llm_api_style,
        )

    def _norm_traits(self, uid: int) -> np.ndarray:
        t = self.user_traits[uid]
        return np.array([t.activity / 3.0, t.diversity / 3.0, t.conformity / 3.0], dtype=np.float32)

    def _compute_exit_threshold(self, uid: int) -> float:
        t = self.user_traits[uid]
        base = {1.0: 3.0, 2.0: 4.6, 3.0: 6.0}.get(float(t.activity), 4.4)
        return self.exit_threshold_scale * (base + 0.25 * (t.diversity - 1.0))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-8 or nb <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    def _action_to_weights(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action.astype(np.float32), -1.0, 1.0)
        w = np.exp(action)
        w = w / np.clip(w.sum(), 1e-6, None)
        return w

    def _oracle_behavior_target(self, repetition: float, uid: int) -> np.ndarray:
        traits = self._norm_traits(uid)
        target = np.array(
            [
                0.55 + 0.30 * traits[0],
                0.20 + 0.35 * traits[1],
                0.25 + 0.50 * repetition,
            ],
            dtype=np.float32,
        )
        target = target / np.clip(target.sum(), 1e-6, None)
        return np.clip(target * 2.0 - 1.0, -1.0, 1.0)

    @staticmethod
    def _normalize_field(value) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().strip(";").split())

    def _split_watch_titles(self, watch_text: str, candidate_titles: List[str]) -> List[str]:
        watch_text = self._normalize_field(watch_text)
        if not watch_text:
            return []
        if watch_text.lower() in {"none", "no", "n/a", "null", "[]"}:
            return []

        if candidate_titles:
            lowered_watch = watch_text.lower()
            matched = []
            for title in sorted(candidate_titles, key=len, reverse=True):
                if title.lower() in lowered_watch:
                    matched.append(title)
            if matched:
                matched_set = set(matched)
                return [title for title in candidate_titles if title in matched_set]

        for token in ["|", "/", ";", " and "]:
            watch_text = watch_text.replace(token, ",")
        return [self._normalize_field(x) for x in watch_text.split(",") if self._normalize_field(x)]

    def _parse_reaction_output(self, response: str, candidate_titles: List[str]) -> Dict[str, List[Dict[str, str]]]:
        import re

        text = response or ""
        align_entries: List[Dict[str, str]] = []
        rating_entries: List[Dict[str, str]] = []
        watch_titles: List[str] = []
        watch_reason = ""

        align_pattern = re.compile(
            r"^\s*MOVIE\s*:\s*(.+?)\s*;\s*ALIGN\s*:\s*([^;\n]+)\s*;\s*REASON\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        rating_pattern = re.compile(
            r"^\s*MOVIE\s*:\s*(.+?)\s*;\s*RATING\s*:\s*(\d+)\s*;\s*FEELING\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        watch_pattern = re.compile(
            r"^\s*NUM\s*:\s*\d+\s*;\s*WATCH\s*:\s*(.*?)\s*;\s*REASON\s*:\s*(.*?)\s*;?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        watch_pattern_loose = re.compile(
            r"^\s*WATCH\s*:\s*(.*?)\s*;\s*REASON\s*:\s*(.*?)\s*;?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )

        for match in align_pattern.finditer(text):
            align_entries.append(
                {
                    "title": self._normalize_field(match.group(1)),
                    "align": self._normalize_field(match.group(2)).lower(),
                    "reason": self._normalize_field(match.group(3)),
                }
            )

        for match in rating_pattern.finditer(text):
            try:
                rating_num = int(self._normalize_field(match.group(2)))
            except ValueError:
                continue
            rating_entries.append(
                {
                    "title": self._normalize_field(match.group(1)),
                    "rating": rating_num,
                    "feeling": self._normalize_field(match.group(3)),
                }
            )

        watch_match = watch_pattern.search(text)
        if not watch_match:
            watch_match = watch_pattern_loose.search(text)
        if watch_match:
            watch_titles = self._split_watch_titles(watch_match.group(1), candidate_titles)
            watch_reason = self._normalize_field(watch_match.group(2))

        watch_titles = list(dict.fromkeys([self._normalize_field(t) for t in watch_titles if t]))
        return {
            "align_entries": align_entries,
            "rating_entries": rating_entries,
            "watch_titles": watch_titles,
            "watch_reason": watch_reason,
        }

    def _get_state(self) -> np.ndarray:
        trait_vec = self._norm_traits(self.current_user)
        session = np.array(
            [
                self.page_idx / max(self.max_pages, 1),
                self.dissatisfaction / max(self.exit_threshold, 1e-6),
                self.last_genre_repetition,
                self.recent_satisfaction,
            ],
            dtype=np.float32,
        )
        state = np.concatenate(
            [
                self.user_pref[self.current_user],
                trait_vec,
                session,
                self.recent_genre,
            ],
            axis=0,
        )
        return state.astype(np.float32)

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            self.current_user = int(self.rng.choice(self.user_ids))
        else:
            self.current_user = int(user_id)
        if self.current_user not in self.user_traits:
            raise ValueError(f"user_id {self.current_user} not found in user traits")

        self.page_idx = 0
        self.dissatisfaction = 0.0
        self.exit_threshold = self._compute_exit_threshold(self.current_user)
        self.recent_genre = np.zeros(self.genre_dim, dtype=np.float32)
        self.last_genre_repetition = 0.0
        self.recent_satisfaction = 0.5
        self.shown_mask = np.zeros(self.n_items, dtype=bool)
        self.session_likes = 0
        self.session_shown = 0
        self.session_satisfaction_sum = 0.0

        init_property = self.persona_df.iloc[self.current_user]
        init_statistic = self.user_stat_df.loc[self.current_user]
        self.avatar = Avatar(
            args=self._build_avatar_args(),
            avatar_id=int(self.current_user),
            init_property=init_property,
            init_statistic=init_statistic,
        )
        self.avatar.exit_flag = False
        self.avatar.negative_feedback_count = 0

        self.current_observation = self._get_state()
        return self.current_observation.copy()

    def _build_recommendation_text(self, selected: np.ndarray) -> Tuple[str, List[str]]:
        lines = []
        titles = []
        for iid in selected.tolist():
            title = self.item_title[iid] if self.item_title[iid] else f"Movie {iid}"
            summary = self.item_summary[iid] if self.item_summary[iid] else "No summary."
            rating = float(self.item_quality[iid]) * 4.0 + 1.0
            lines.append(
                f"<- {title} -> <- History ratings: {rating:.2f} -> <- Summary: {summary} ->\n"
            )
            titles.append(title)
        return "".join(lines), titles

    def _extract_interview_rating(self) -> Optional[float]:
        if self.avatar is None:
            return None
        prompt = (
            "Do you feel satisfied with the recommender system you have just interacted? "
            "Rate this recommender system from 1-10 and explain briefly. "
            "Use format: RATING: [integer between 1 and 10]; REASON: [brief reason]"
        )
        try:
            response = self.avatar.response_to_question(prompt, remember=False)
        except Exception:
            return None
        import re

        match = re.search(r"RATING\s*:\s*(\d+)", response or "", flags=re.IGNORECASE)
        if not match:
            return None
        try:
            score = float(int(match.group(1)))
        except Exception:
            return None
        return float(np.clip(score, 1.0, 10.0))

    def _compute_terminal_score(self, done_reason: str) -> float:
        session_len = max(int(self.page_idx), 1)
        like_rate = float(self.session_likes) / max(float(self.session_shown), 1.0)
        avg_satisfaction = float(self.session_satisfaction_sum) / float(session_len)
        sat01 = float(np.clip((avg_satisfaction + 1.0) * 0.5, 0.0, 1.0))
        len01 = float(np.clip(session_len / max(float(self.max_pages), 1.0), 0.0, 1.0))

        score = self.terminal_score_max * (0.35 * like_rate + 0.35 * sat01 + 0.30 * len01)
        if done_reason in {"avatar_exit", "dissatisfaction_proxy"}:
            score -= 0.8
        elif done_reason == "max_pages":
            score += 0.2
        elif done_reason == "exhausted":
            score -= 0.2
        return float(np.clip(score, 0.0, self.terminal_score_max))

    def step(self, action: np.ndarray):
        if self.avatar is None:
            raise RuntimeError("Environment is not reset.")

        self.page_idx += 1
        traits = self._norm_traits(self.current_user)
        activity_norm = float(traits[0])

        w = self._action_to_weights(action)
        user_pref = self.user_pref[self.current_user]

        available = np.where(~self.shown_mask)[0]
        if len(available) == 0:
            done_reason = "no_item"
            terminal_score = self._compute_terminal_score(done_reason)
            info = {
                "likes": 0,
                "shown": 0,
                "genre_repetition": self.last_genre_repetition,
                "satisfaction": self.recent_satisfaction,
                "done_reason": done_reason,
                "terminal_score": terminal_score,
                "behavior_target_action": self._oracle_behavior_target(self.last_genre_repetition, self.current_user),
                "effect_action_feature": np.zeros(self.genre_dim, dtype=np.float32),
                "immediate_response": np.zeros((0, self.response_dim), dtype=np.float32),
            }
            self.current_observation = self._get_state()
            return self.current_observation.copy(), -0.2, True, info

        item_genre = self.item_genre[available]
        relevance = item_genre @ user_pref
        novelty = self.item_novelty[available]

        if np.linalg.norm(self.recent_genre) > 1e-8:
            recent_rep = np.array([self._cosine(g, self.recent_genre) for g in item_genre], dtype=np.float32)
            diversity_bonus = 1.0 - recent_rep
        else:
            diversity_bonus = np.ones(len(available), dtype=np.float32) * 0.5

        score = w[0] * relevance + w[1] * novelty + w[2] * diversity_bonus
        score += self.rng.normal(0.0, 0.01, size=score.shape)

        top_k = min(self.slate_size, len(available))
        top_local_idx = np.argpartition(-score, top_k - 1)[:top_k]
        local_scores = score[top_local_idx]
        order = np.argsort(-local_scores)
        top_local_idx = top_local_idx[order]
        selected = available[top_local_idx]
        self.shown_mask[selected] = True

        rec_text, selected_titles = self._build_recommendation_text(selected)
        reaction = self.avatar.reaction_to_recommended_items(rec_text, self.page_idx)
        parsed = self._parse_reaction_output(reaction, selected_titles)

        align_map: Dict[str, float] = {}
        rating_map: Dict[str, int] = {}
        watch_set = {self._normalize_field(t).lower() for t in parsed["watch_titles"]}

        for entry in parsed["align_entries"]:
            key = self._normalize_field(entry["title"]).lower()
            if not key:
                continue
            align_map[key] = 1.0 if entry["align"] in {"yes", "y", "true"} else 0.0
        for entry in parsed["rating_entries"]:
            key = self._normalize_field(entry["title"]).lower()
            if not key:
                continue
            rating_map[key] = int(entry["rating"])

        immediate = np.zeros((top_k, self.response_dim), dtype=np.float32)
        for i, title in enumerate(selected_titles):
            key = self._normalize_field(title).lower()
            rating = float(rating_map.get(key, 0))
            watch = 1.0 if key in watch_set else 0.0
            align = float(align_map.get(key, 0.0))
            if key not in align_map and watch > 0:
                align = 1.0
            like = 1.0 if (watch > 0 and rating >= 4) else 0.0
            dislike = 1.0 if (align <= 0.0 or (watch > 0 and rating > 0 and rating <= 2)) else 0.0
            immediate[i] = np.array([watch, like, align, dislike], dtype=np.float32)

        watch_ratio = float(immediate[:, 0].mean()) if top_k > 0 else 0.0
        like_ratio = float(immediate[:, 1].mean()) if top_k > 0 else 0.0
        align_ratio = float(immediate[:, 2].mean()) if top_k > 0 else 0.0
        dislike_ratio = float(immediate[:, 3].mean()) if top_k > 0 else 0.0

        selected_genre = self.item_genre[selected]
        current_genre = selected_genre.mean(axis=0)
        genre_rep = self._cosine(current_genre, self.recent_genre)

        satisfaction = (
            0.45 * align_ratio
            + 0.35 * like_ratio
            + 0.20 * watch_ratio
            - self.repetition_penalty_scale * self.repeat_aversion_scale * 0.35 * genre_rep
            - 0.35 * dislike_ratio
        )
        satisfaction = float(np.clip(satisfaction, -1.0, 1.0))

        fatigue = self.fatigue_scale * (0.02 + 0.04 * (1.0 - activity_norm))
        self.dissatisfaction += max(0.0, 0.45 - satisfaction)
        self.dissatisfaction += (
            self.repetition_penalty_scale * self.repeat_aversion_scale * 0.45 * genre_rep + fatigue
        )
        self.dissatisfaction -= max(0.0, satisfaction - 0.55) * 0.20
        self.dissatisfaction = max(0.0, self.dissatisfaction)

        self.recent_genre = 0.65 * self.recent_genre + 0.35 * current_genre
        gsum = float(self.recent_genre.sum())
        if gsum > 1e-8:
            self.recent_genre = self.recent_genre / gsum
        self.last_genre_repetition = float(genre_rep)
        self.recent_satisfaction = 0.70 * self.recent_satisfaction + 0.30 * satisfaction

        likes_cnt = int(np.sum(immediate[:, 1]))
        self.session_likes += likes_cnt
        self.session_shown += int(top_k)
        self.session_satisfaction_sum += satisfaction

        reward = (
            1.10 * like_ratio
            + 0.40 * watch_ratio
            + 0.55 * satisfaction
            - self.repetition_penalty_scale * self.repeat_aversion_scale * 0.30 * genre_rep
            - 0.03 * self.page_idx
        )

        done = False
        done_reason = ""
        if self.page_idx >= self.max_pages:
            done = True
            done_reason = "max_pages"
        elif bool(self.avatar.exit_flag):
            done = True
            done_reason = "avatar_exit"
        elif np.all(self.shown_mask):
            done = True
            done_reason = "exhausted"
        elif self.dissatisfaction >= (1.2 * self.exit_threshold):
            done = True
            done_reason = "dissatisfaction_proxy"

        if not done:
            reward += 0.03

        terminal_score = None
        if done:
            terminal_score = self._compute_terminal_score(done_reason)
            if self.terminal_from_interview:
                interview_score = self._extract_interview_rating()
                if interview_score is not None:
                    terminal_score = interview_score

        info = {
            "likes": likes_cnt,
            "shown": int(top_k),
            "genre_repetition": float(genre_rep),
            "satisfaction": float(satisfaction),
            "done_reason": done_reason,
            "terminal_score": terminal_score,
            "behavior_target_action": self._oracle_behavior_target(float(genre_rep), self.current_user),
            "effect_action_feature": current_genre.astype(np.float32),
            "immediate_response": immediate,
        }

        self.current_observation = self._get_state()
        return self.current_observation.copy(), float(reward), bool(done), info
