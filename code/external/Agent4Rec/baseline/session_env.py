from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class UserTraits:
    activity: float
    diversity: float
    conformity: float


class LongSessionRecEnv:
    """
    Lightweight long-session simulator for Agent4Rec dataset.

    Design goals:
    - run RL baselines efficiently (no LLM in the loop)
    - keep long-session dynamics explicit (fatigue + dissatisfaction + repetition)
    - expose "exit threshold" to control average session length
    """

    def __init__(
        self,
        root_dir: str,
        dataset: str = "ml-1m",
        max_pages: int = 20,
        slate_size: int = 6,
        seed: int = 11,
        user_limit: Optional[int] = None,
        exit_threshold_scale: float = 1.35,
        repetition_penalty_scale: float = 1.20,
        repeat_aversion_scale: float = 1.50,
        fatigue_scale: float = 1.80,
        terminal_score_max: float = 10.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dataset = str(dataset)
        self.max_pages = int(max_pages)
        self.slate_size = int(slate_size)
        self.exit_threshold_scale = float(exit_threshold_scale)
        self.repetition_penalty_scale = float(repetition_penalty_scale)
        self.repeat_aversion_scale = float(repeat_aversion_scale)
        self.fatigue_scale = float(fatigue_scale)
        self.terminal_score_max = float(terminal_score_max)
        self.rng = np.random.default_rng(seed)

        self._load_data(user_limit=user_limit)

        self.action_dim = 3
        self.effect_dim = self.genre_dim
        self.state_dim = self.genre_dim + 3 + 4 + self.genre_dim

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

    @staticmethod
    def _tokenize_genres(raw_value: object) -> List[str]:
        tokens: List[str] = []
        for token in str(raw_value).split("|"):
            token = token.strip()
            if not token:
                continue
            # Book metadata may append per-item author tags into the genre field.
            # Keeping them explodes the vocabulary and breaks RL environment memory use.
            if token.lower().startswith("author:"):
                continue
            tokens.append(token)
        return tokens

    def _load_data(self, user_limit: Optional[int]) -> None:
        sim_dir = self.root_dir / "datasets" / self.dataset / "simulation"
        cf_dir = self.root_dir / "datasets" / self.dataset / "cf_data"

        movie_df = pd.read_csv(sim_dir / "movie_detail.csv")
        movie_df = movie_df.sort_values("movie_id").reset_index(drop=True)
        self.n_items = int(movie_df["movie_id"].max()) + 1

        tokenized_genres: List[List[str]] = []
        genre_freq: Dict[str, int] = {}
        for raw_genres in movie_df["genres"].fillna(""):
            tokens = self._tokenize_genres(raw_genres)
            tokenized_genres.append(tokens)
            for token in set(tokens):
                genre_freq[token] = genre_freq.get(token, 0) + 1

        max_genre_vocab = 256
        use_other_bucket = False
        if len(genre_freq) > max_genre_vocab:
            top_tokens = sorted(
                genre_freq.items(),
                key=lambda kv: (-kv[1], kv[0]),
            )[: max_genre_vocab - 1]
            self.genre_list = [token for token, _ in top_tokens]
            use_other_bucket = True
            self.genre_list.append("__other__")
        else:
            self.genre_list = sorted(genre_freq)
        self.genre2idx = {g: i for i, g in enumerate(self.genre_list)}
        self.genre_dim = len(self.genre_list)

        item_genre = np.zeros((self.n_items, self.genre_dim), dtype=np.float32)
        item_quality = np.zeros(self.n_items, dtype=np.float32)

        for row_idx, row in movie_df.iterrows():
            iid = int(row["movie_id"])
            item_quality[iid] = float(row["rating"])
            assigned = False
            for g in tokenized_genres[row_idx]:
                if g in self.genre2idx:
                    item_genre[iid, self.genre2idx[g]] = 1.0
                    assigned = True
            if use_other_bucket and tokenized_genres[row_idx] and not assigned:
                item_genre[iid, self.genre2idx["__other__"]] = 1.0

        quality_min = float(item_quality.min())
        quality_max = float(item_quality.max())
        self.item_quality = (item_quality - quality_min) / max(quality_max - quality_min, 1e-6)
        self.item_genre = item_genre

        # popularity from train interactions
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
        if user_limit is not None:
            user_ids = user_ids[: int(user_limit)]

        self.user_ids = np.array(user_ids, dtype=np.int32)
        self.user_pref = user_pref
        self.user_traits = user_traits

    def _norm_traits(self, uid: int) -> np.ndarray:
        t = self.user_traits[uid]
        return np.array([t.activity / 3.0, t.diversity / 3.0, t.conformity / 3.0], dtype=np.float32)

    def _compute_exit_threshold(self, uid: int) -> float:
        t = self.user_traits[uid]
        base = {1.0: 3.2, 2.0: 4.8, 3.0: 6.2}.get(float(t.activity), 4.5)
        # higher diversity users tolerate broader exploration slightly longer
        return self.exit_threshold_scale * (base + 0.3 * (t.diversity - 1.0))

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
        # heuristic target inspired by KuaiSim HAC's extra behavior guidance
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

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            self.current_user = int(self.rng.choice(self.user_ids))
        else:
            self.current_user = int(user_id)

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

        return self._get_state()

    def _compute_terminal_score(self, done_reason: str) -> float:
        session_len = max(int(self.page_idx), 1)
        like_rate = float(self.session_likes) / max(float(self.session_shown), 1.0)
        avg_satisfaction = float(self.session_satisfaction_sum) / float(session_len)
        sat01 = float(np.clip((avg_satisfaction + 1.0) * 0.5, 0.0, 1.0))
        len01 = float(np.clip(session_len / max(float(self.max_pages), 1.0), 0.0, 1.0))

        score = self.terminal_score_max * (
            0.35 * like_rate
            + 0.35 * sat01
            + 0.30 * len01
        )

        if done_reason == "dissatisfaction":
            score -= 1.0
        elif done_reason == "max_pages":
            score += 0.3
        elif done_reason == "exhausted":
            score -= 0.3

        return float(np.clip(score, 0.0, self.terminal_score_max))

    def step(self, action: np.ndarray):
        self.page_idx += 1
        traits = self._norm_traits(self.current_user)
        activity_norm = float(traits[0])
        conformity_norm = float(traits[2])

        w = self._action_to_weights(action)
        user_pref = self.user_pref[self.current_user]

        available = np.where(~self.shown_mask)[0]
        if len(available) == 0:
            info = {
                "likes": 0,
                "shown": 0,
                "genre_repetition": self.last_genre_repetition,
                "satisfaction": self.recent_satisfaction,
                "done_reason": "no_item",
                "terminal_score": self._compute_terminal_score("no_item"),
                "behavior_target_action": self._oracle_behavior_target(self.last_genre_repetition, self.current_user),
                "effect_action_feature": np.zeros(self.genre_dim, dtype=np.float32),
            }
            return self._get_state(), -0.2, True, info

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
        selected = available[top_local_idx]

        self.shown_mask[selected] = True

        selected_genre = self.item_genre[selected]
        selected_rel = relevance[top_local_idx]
        selected_nov = novelty[top_local_idx]
        selected_quality = self.item_quality[selected]

        current_genre = selected_genre.mean(axis=0)
        genre_rep = self._cosine(current_genre, self.recent_genre)
        page_rel = float(selected_rel.mean()) if len(selected_rel) else 0.0
        page_nov = float(selected_nov.mean()) if len(selected_nov) else 0.0
        page_div = 1.0 - genre_rep

        p_watch = (
            0.05
            + 0.35 * activity_norm
            + 0.55 * selected_rel
            + 0.05 * selected_nov
            - self.repetition_penalty_scale * self.repeat_aversion_scale * 0.35 * genre_rep
        )
        p_watch = np.clip(p_watch, 0.01, 0.99)
        watched = self.rng.random(top_k) < p_watch

        p_like = 0.05 + 0.65 * selected_rel + 0.10 * conformity_norm * selected_quality + 0.10 * selected_nov
        p_like = np.clip(p_like, 0.01, 0.99)
        liked = watched & (self.rng.random(top_k) < p_like)

        watched_cnt = int(watched.sum())
        liked_cnt = int(liked.sum())

        like_ratio = liked_cnt / max(top_k, 1)
        watch_ratio = watched_cnt / max(top_k, 1)

        self.session_likes += liked_cnt
        self.session_shown += int(top_k)

        satisfaction = (
            0.55 * page_rel
            + 0.25 * page_nov
            + 0.20 * page_div
            + 0.15 * like_ratio
            - self.repetition_penalty_scale * self.repeat_aversion_scale * 0.35 * genre_rep
        )
        if watched_cnt == 0:
            satisfaction -= 0.08
        satisfaction = float(np.clip(satisfaction, -1.0, 1.0))
        self.session_satisfaction_sum += satisfaction

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

        reward = (
            1.10 * like_ratio
            + 0.35 * watch_ratio
            + 0.65 * satisfaction
            - self.repetition_penalty_scale * self.repeat_aversion_scale * 0.35 * genre_rep
            - 0.02 * self.page_idx
        )

        done_reason = ""
        done = False
        if self.page_idx >= self.max_pages:
            done = True
            done_reason = "max_pages"
        elif self.dissatisfaction >= self.exit_threshold:
            done = True
            done_reason = "dissatisfaction"
        elif np.all(self.shown_mask):
            done = True
            done_reason = "exhausted"

        if not done:
            reward += 0.05

        terminal_score = None
        if done:
            terminal_score = self._compute_terminal_score(done_reason)

        info = {
            "likes": liked_cnt,
            "shown": top_k,
            "genre_repetition": float(genre_rep),
            "satisfaction": float(satisfaction),
            "done_reason": done_reason,
            "terminal_score": terminal_score,
            "behavior_target_action": self._oracle_behavior_target(float(genre_rep), self.current_user),
            "effect_action_feature": current_genre.astype(np.float32),
        }

        return self._get_state(), float(reward), bool(done), info
