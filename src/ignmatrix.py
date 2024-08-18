from abc import ABC, abstractmethod
from dataclasses import dataclass
from Levenshtein import ratio as leven_ratio
from typing import Optional
from utils import ndefault
from enums import IGNMatrixMode


__all__ = [
    "Player",
    "IGNMatrix",
]


@dataclass
class Player:
    """Dataclass containing all necessary player information"""
    idx: int
    ign: str
    team: int | None = None


class IGNMatrix(ABC):
    """
    IGN Matrix infers the true IGN from the EasyOCR reading of the IGN (pseudoIGN/pign) from the killfeed.
    The matrix can be initialised with a prior list of 'fixed' IGNs (IGNs known before starting the game)
    If the matrix is not provided with 10 fixed IGNs, the matrix's output will depend on its initialised mode
    - fixed: will return None for all non-fixed IGNs
    - infer: will infer the non-fixed IGNs from the OCR's output
    The matrix will return an index/ID and the true/most-seen IGN when requested using the `get` method
    The matrix uses Levenshtein distance to determine whether two IGNs are the same (could be improved?)
    """

    @property
    @abstractmethod
    def mode(self) -> IGNMatrixMode:
        ...

    @abstractmethod
    def get(self, pseudoIGN: str, threshold: float = 0.75) -> Optional[Player]:
        ...
    
    @abstractmethod
    def get_from_idx(self, idx: int) -> Optional[Player]:
        ...
    
    @abstractmethod
    def get_team_from_idx(self, idx: int) -> Optional[int]:
        ...
    
    @abstractmethod
    def get_players(self) -> list[Player]:
        """Returns a list of 10 Player objects where each team are grouped in team 0: 0-4 | team 1: 5-9"""
        ...
    
    @abstractmethod
    def evaluate(self, pign: str) -> float:
        ...
    
    def update_team_table(self, pidx: int, tidx: int) -> None:
        ...
    
    @staticmethod
    def new(igns: list[str], mode: IGNMatrixMode) -> 'IGNMatrix':
        """Creates a new IGNMatrix object from a list of fixed IGNs and the IGNMatrix Mode"""
        match mode:
            case IGNMatrixMode.FIXED:
                return IGNMatrixFixed.new(igns)
            case IGNMatrixMode.INFER:
                if len(igns) == 10: return IGNMatrixFixed.new(igns)
                else: return IGNMatrixInfer.new(igns)
            case _:
                raise ValueError(f"Unknown IGNMatrixMode {mode}")

    @staticmethod
    def compare_names(name1: str, name2: str) -> float:
        """Compares two IGN's (pseudo/non-pseudo) using Levenshtein distance, output in the range [0-1]"""
        return leven_ratio(name1.lower(), name2.lower())
    
    @staticmethod
    def _check_fixed(ign_list: list[str], pign: str, threshold: float) -> Optional[Player]:
        """Compares a psuedoIGN to a fixed list of igns and returns a Player object if the pign is in the list"""
        max_score = 0.0
        max_idx   = None
        for i, ign in enumerate(ign_list):
            if ign == pign: return Player(i, ign, int(i >= 5))
            if (score := IGNMatrix.compare_names(pign, ign)) > max_score:
                max_score = score
                max_idx = i

        if max_score > threshold and max_idx is not None:
            return Player(max_idx, ign_list[max_idx], int(max_idx >= 5))
        return None


class IGNMatrixFixed(IGNMatrix):
    """
    This subclass of IGNMatrix only handles the case where all IGNs are known beforehand
    Separating the different IGN modes aims to improve efficiency, readability and modularity
    """
    VALID_THRESHOLD = 0.6 ## threshold for Levenshtein distance to determine equality

    def __init__(self, igns: list[str]) -> None:
        self.__fixmat = igns
    
    @property
    def mode(self) -> IGNMatrixMode:
        return IGNMatrixMode.FIXED
    
    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Optional[Player]:
        """Returns the most-likely Player object from a pseudoIGN or None if the pseudoIGN does not meet the threshold"""
        return IGNMatrix._check_fixed(self.__fixmat, pign, threshold)

    def get_from_idx(self, idx: int) -> Optional[Player]:
        """Returns the Player object from their index"""
        if 0 <= idx <= 9:
            return Player(idx, self.__fixmat[idx], int(idx >= 5))

        return None
    
    def get_team_from_idx(self, idx: int) -> Optional[int]:
        """Returns the team index from the player's index"""
        return int(idx >= 5)

    def get_players(self) -> list[Player]:
        """Returns a list of all Players currently playing"""
        return [Player(i, ign, int(i >= 5)) for i, ign in enumerate(self.__fixmat)]

    ## worth adding an cache?
    def evaluate(self, pign: str) -> float:
        """Evaluates a given pseudoIGN and returns as score between 0-1"""
        if pign in self.__fixmat: return 1.0
        return max([IGNMatrix.compare_names(pign, name) for name in self.__fixmat])

    @staticmethod
    def new(igns: list[str]) -> 'IGNMatrixFixed':
        """Type checking of `igns` is done by the `__cparse_IGNS` function, except for length=10 check"""
        if len(igns) != 10:
            raise ValueError(f"Invalid Fixed IGNMatrix argument, must have 10 IGNs, not {len(igns)}")
        
        return IGNMatrixFixed(igns)


class NamesRecord:
    """A Record containing the occurrency count for each pseudoIGN of an unknown true IGN"""
    def __init__(self, pign: str):
        self.__vars:   list[str] = [pign]
        self.__counts: list[int] = [1]

        self.__noccr = 1
        self.__best = 0 ## idx of most-likely pign so far

    def inc(self, idx: int) -> None:
        self.__counts[idx] += 1
        self.__noccr += 1

        if self.__counts[idx] > self.__counts[self.__best]:
            self.__best = self.__counts[idx]

    def new(self, pign: str) -> None:
        self.__vars.append(pign)
        self.__counts.append(1)
        self.__noccr += 1

    def _in(self, pign: str) -> int:
        if pign in self.__vars:
            return self.__vars.index(pign)
        return -1

    def fuzzy_cmp(self, pign: str, threshold: float) -> bool:
        for var in self.__vars:
            if IGNMatrix.compare_names(pign, var) >= threshold:
                return True
        return False

    def noccr(self) -> int:
        return self.__noccr

    def maxoccr(self) -> int:
        return max(self.__counts)

    def best(self) -> str:
        return self.__vars[self.__best]

    def __iter__(self):
        return iter(zip(self.__vars, self.__counts))


class IGNMatrixInfer(IGNMatrix):
    """
    IGN Inference has 3 steps, fixed, semi-fixed, matrix
    1. Fixed      - a pseudoIGN is compared to the list of known IGNs
    2. Semi-Fixed - a pseudoIGN is tested against all of the semi-fixed IGN names-record
    3. Matrix     - a pseudoIGN is tested against all other pseudoIGN names-record
    When a pseudoIGN is found in either semi-fixed or matrix names-record, the occurrence of that pseudoIGN is incremented
    If a matrix names-record reaches the ASSIMILATION THRESHOLD of occurrences, that pseduo IGN names-record is promoted to semi-fixed
    Once len(Fixed) + len(Semi-Fixed) == 10, no more psuedoIGN names-record can be promoted and assumed to be invalid IGNs
    """
    VALID_THRESHOLD = 0.75     ## threshold for Levenshtein distance to determine equality
    ASSIMILATE_THRESHOLD = 7   ## how many times a pseudoIGN has to be seen before adding to 'semi-fixed' list
    KD_DIFF_THRESHOLD = 4      ## how many times player on team 'A' but kill/be killed by a player on team 'B' before setting team
    NO_TEAM_HIST_THRESHOLD = 3 ## how many history records to declare teams, if no team indices are known
    EVAL_DECAY = 0.95

    def __init__(self, igns: list[str]) -> None:
        self.__fixmat = igns
        self.__fixlen = len(igns)

        self.__semi_fixmat: dict[int, NamesRecord] = {}
        self.__matrix:      dict[int, NamesRecord] = {}
        self.__mats = (self.__semi_fixmat, self.__matrix)
        self.__semi_fixlen = 0
        self.__idx_counter = self.__fixlen

        self.__team_mat:  dict[int, int]       = {i: int(i>=5) for i in range(self.__fixlen)}
        self.__team_diff: dict[int, list[int]] = {}  ## for each idx, number of kills/deaths to either team
        self.__team_hist: dict[int, list[int]] = {}  ## a list of kill/death relations for each idx

    @property
    def mode(self) -> IGNMatrixMode:
        return IGNMatrixMode.INFER

    def get(self, pign: str, threshold: float = VALID_THRESHOLD) -> Optional[Player]:
        """
        Returns the most-likely Player from the pseudoIGN
        1. checks the pign against the fixed matrix
        2. checks the pign against the semi-fixed matrix
        3. checks the pign against the matrix & assimmilates player into semi-fixed if assim-threshold is met
        4. otherwise add pign to the matrix
        """
        ## first check the fixed igns
        if self.__fixlen > 0 and (pl := IGNMatrix._check_fixed(self.__fixmat, pign, threshold)) is not None:
            return pl

        ## second, check the semi-fixed igns
        for idx, nr in self.__semi_fixmat.items():
            if (pl := self.__in_names_record(pign, idx, nr, threshold)):
                return pl

        ## if all fixed and semi-fixed igns have been found, assume pseudoIGN is invalid
        if self.__fixlen + self.__semi_fixlen >= 10:
            return None

        ## if not all semi-fixed igns have been found, check matrix, and assimilate if necessary
        for idx, nr in self.__matrix.items():
            if (pl := self.__in_names_record(pign, idx, nr, threshold)):
                self.__check_assimilation(idx, nr)
                return pl

        ## if ign has not been seen, add to matrix
        idx = self.__idx_counter
        self.__matrix[idx] = NamesRecord(pign)
        self.__idx_counter += 1

        return Player(idx, pign, None)

    def __in_names_record(self, pign: str, idx: int, nr: NamesRecord, threshold: float) -> Optional[Player]:
        """
        Determines whether a pseudoIGN/pign belongs to a specified NamesRecord,
          and increases the occurrence counts if it does.
        """
        if (pidx := nr._in(pign)) >= 0:
            nr.inc(pidx) # increase occurrence count for pseudoIGN
            return Player(idx, nr.best(), self.get_team_from_idx(idx))

        if nr.fuzzy_cmp(pign, threshold):
            nr.new(pign)
            return Player(idx, nr.best(), self.get_team_from_idx(idx))

        return None

    def __check_assimilation(self, idx: int, nr: NamesRecord) -> None:
        """Checks to see if a specified names record has passed over the Assimilation threshold, and if so assimilate to semi-fixed matrix"""
        if nr.noccr() >= IGNMatrixInfer.ASSIMILATE_THRESHOLD:
            self.__semi_fixmat[idx] = self.__matrix.pop(idx)
            self.__semi_fixlen += 1

    def get_ign_from_idx(self, idx: int) -> Optional[str]:
        """Returns the player's most-likely ign from their index"""
        if 0 <= idx < self.__fixlen:
            return self.__fixmat[idx]
        elif idx in self.__semi_fixmat:
            return self.__semi_fixmat[idx].best()
        elif idx in self.__matrix:
            return self.__matrix[idx].best()

        return None

    def get_from_idx(self, idx: int) -> Optional[Player]:
        """Returns the (most likely) Player from a given idx"""
        if (ign := self.get_ign_from_idx(idx)) is None:
            return None

        return Player(idx, ign, self.get_team_from_idx(idx))

    def get_team_from_idx(self, idx: int) -> Optional[int]:
        """Returns the team index from a player's index"""
        if self.__fixlen >= 5 or idx <= self.__fixlen:
            return int(idx >= 5)

        if idx in self.__team_mat:
            return self.__team_mat[idx]

        return None

    def update_team_table(self, pidx: int, tidx: int) -> None:
        """
        To infer a player's team, a record of player kills/deaths are recorded to infer the most likely team they belong to
        The team table is required to handle cases of team-kills, and so the KD_DIFF_THRESHOLD is used to determine 
          when a player has had enough interactions to decide a team
        """
        pteam = self.__team_mat.get(pidx, None)
        tteam = self.__team_mat.get(tidx, None)

        ## parentheses for clarity
        if ((pteam is not None) and (tteam is not None)) \
            or (self.get_team_from_idx(pidx) is not None and self.get_team_from_idx(tidx) is not None): ## both team indices are known
            return

        ## one of the team indices are not known
        elif (pteam is None) and (tteam is not None):
            self.__add_team_diff(pidx, tteam)
        elif pteam is not None and (tteam is None):
            self.__add_team_diff(tidx, pteam)

        ## neither team indices are known
        else:
            self.__add_team_hist(pidx, tidx)
            self.__add_team_hist(tidx, pidx)

    def __add_team_diff(self, idx: int, opp_team: int) -> None:
        """
        This method adds an interaction count with `opp_team` to idx's team_diff
        It then checks to see if `idx`'s team_diff has reached the KD_DIFF_TRESHOLD
          so that idx's team can be set
        """
        if idx not in self.__team_diff:
            self.__team_diff[idx] = [0, 0]

        tdiff = self.__team_diff[idx]
        tdiff[opp_team] += 1

        ## check if idx has passed KD_DIFF_THRESHOLD, number of interactions before setting team index
        if abs(tdiff[0] - tdiff[1]) > IGNMatrixInfer.KD_DIFF_THRESHOLD:
            team_idx = int(tdiff[0] > tdiff[1])      ## idx's team = argmin(tdiff)
            self.__set_team_idx(idx, team_idx)

    def __add_team_hist(self, idx: int, value: int) -> None:
        """In the case when neither player's team in an interaction is known, add the interaction to a history"""
        if idx in self.__team_mat: return  ## should only hit if I've made a mistake

        if idx not in self.__team_hist:
            self.__team_hist[idx] = [value]
        else:
            self.__team_hist[idx].append(value)

        ## in the case when no player has a set team, declare a team when the # interactions threshold is met
        if len(self.__team_mat) == 0 and len(self.__team_hist[idx]) > IGNMatrixInfer.NO_TEAM_HIST_THRESHOLD:
            self.__set_team_idx(idx, 0)  ## DECLARE TEAMS!

    def __set_team_idx(self, idx: int, team_idx: int) -> None:
        """Sets the team index for a Player with `idx`"""
        self.__team_mat[idx] = team_idx       ## set the team index for idx

        ## use interaction history to infer teams for other players
        if idx in self.__team_hist:
            ## update team_diff for all of idx's previous unknown-team interactions
            for opp_idx in self.__team_hist[idx]:
                self.__add_team_diff(opp_idx, team_idx)
            self.__team_hist.pop(idx)

        if idx in self.__team_diff:
            self.__team_diff.pop(idx)

    def get_players(self) -> list[Player]:
        """Returns a list of Players most likely playing in the game"""
        unsorted_players = self.__get_unsorted_players()
        if self.__fixlen >= 5:
            return unsorted_players

        ## sort the players based on their team, 3rd bucket is for players with an unknown team
        buckets = [[], [], []]
        for pl in unsorted_players:
            bidx = ndefault(pl.team, 2)
            buckets[bidx].append(pl)

        return buckets[0] + buckets[1] + buckets[2]

    def __get_unsorted_players(self) -> list[Player]:
        """
        Creates a list of players with priority starting with fixmat, semi-fixmat and matrix
        The likelihood for players in the matrix is determined by the number of occurrences
        """
        players = [Player(i, ign, int(i >= 5)) for i, ign in enumerate(self.__fixmat)]
        if self.__fixlen == 10:
            return players

        players += [Player(idx, nr.best(), self.get_team_from_idx(idx)) for idx, nr in self.__semi_fixmat.items()]
        if len(players) == 10:
            return players

        if len(players) + len(self.__matrix) <= 10:
            ## Note: may return a list < 10 length
            return players + [Player(idx, nr.best(), self.get_team_from_idx(idx)) for idx, nr in self.__matrix.items()]
        else:
            ## Only add the most likely (most seen) players from __matrix to the players list
            ##   sorted the __matrix by # times seen and only add the top 10-len(players)
            indices = sorted(self.__matrix, key=lambda k: self.__matrix[k].noccr(), reverse=True)[:10-len(players)]
            return players + [Player(idx, self.__matrix[idx].best(), self.get_team_from_idx(idx)) for idx in indices]


    def evaluate(self, pign: str) -> float:
        """
        Evaluates an pseudoIGN, determining how close it is to a real ign
        Eval decay: `(IGNMatrixInfer.EVAL_DECAY ** (max_occr - occr)` factor gives emphasis to pseudoIGNs seen more often
        """
        max_val = 0.0
        if pign in self.__fixmat:
            return 1.0

        for ign in self.__fixmat:
            max_val = max(max_val, IGNMatrix.compare_names(pign, ign))

        for nr in self.__semi_fixmat.values():
            if nr._in(pign):
                return 1.0

        evals = [IGNMatrix.compare_names(pign, name) * (IGNMatrixInfer.EVAL_DECAY ** (nr.maxoccr() - occr)) for mat in self.__mats for nr in mat.values() for name, occr in nr]
        return max(max_val, *evals)

    @staticmethod
    def new(igns: list[str]) -> 'IGNMatrixInfer':
        """Type checking of `igns` is done by the `__cparse_IGNS` function"""
        return IGNMatrixInfer(igns)


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
