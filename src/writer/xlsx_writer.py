from pathlib import Path
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from typing import Any

from config import Config
from history import History, HistoryRound
from ignmatrix import IGNMatrix, Player
from stats import PlayerRoundStats, compile_match_stats, iter_killfeed, is_kost, MatchStats_t, RoundStats_t
from utils import ndefault, str2f, perc_s
from utils.enums import SaveFileType, Team

from .base import Writer
from .utils import get_players, handle_write_errors


__all__ = ["XlsxWriter"]


MATCH_SHEET_HEADERS = [
    ["Statistics"],
    ["Player", "Team Index", "Rounds", "Kills", "Deaths", "KST", "KPR", "SRV", "Hs%", "1vX"]
]

ROUND_SHEET_STATS_HEADERS = [
    ["Statistics"],
    ["Player", "Team Index", "Kills", "Death", "Assissts", "Hs%", "Headshots", "1vX", "Operator", "Traded Kills", "Refragged Kills", "Traded Deaths"]
]

ROUND_SHEET_INFO_HEADERS = [
    ["Round Info"],
    ["Name", "Value", "Time"]
]

ROUND_SHEET_KILLFEED_HEADERS = [
    ["Kill/death feed"],
    ["Player", "Target", "Time", "Trade", "Refrag"]
]

EMPTY_ROW = [[]]

WINNING_TEAM_TEXT = {
    Team.TEAM0: "YOUR TEAM [0]",
    Team.TEAM1: "OPPONENTS [1]",
    Team.UNKNOWN: ""
}

BOOL_MAP = {
    True: "TRUE",
    False: "FALSE"
}


def fmt_s(*args): return str2f(perc_s(*args))


class XlsxWriter(Writer):
    def __init__(self, save_path: Path, config: Config) -> None:
        super(XlsxWriter, self).__init__(SaveFileType.XLSX, save_path, config)
    
    def write(self, history: History, ignmat: IGNMatrix) -> None:
        if not history.is_ready:
            return

        players = get_players(history.get_first_round(), ignmat)
        match_stats = compile_match_stats(history, players)

        xslx_match = self.get_match(match_stats, players)
        sheet_data = xslx_match | self.get_rounds(match_stats, history, ignmat)

        wb = self.create_workbook()
        self.save_workbook(wb, sheet_data)
    
    def get_match(self, match_stats: MatchStats_t, players: list[Player]) -> dict:
        """Returns the data present on the Match sheet, statistics across all rounds"""
        match_data = [self.__aggregate_match_data(match_stats, pl) for pl in players]
        return {"Match": MATCH_SHEET_HEADERS + match_data}

    def __aggregate_match_data(self, match_stats: MatchStats_t, pl: Player) -> list[Any]:
        ## "Player", "Team Index", "Rounds", "Kills", "Deaths", "KST", "KPR", "SRV", "Hs%", "1vX"
        prs = [rs[pl.uid] for rs in match_stats if pl.uid in rs]

        rnds   = len(prs)
        kills  = sum([r.kills for r in prs])
        deaths = sum([r.death for r in prs])
        kost   = sum(map(is_kost, prs))
        hs     = sum([r.headshots for r in prs])
        onevx  = sum([r.onevx > 0 for r in prs])
        return [pl.ign, pl.team.value, rnds, kills, deaths,
                fmt_s(kost,rnds), fmt_s(kills,rnds), fmt_s(1-deaths,rnds), fmt_s(hs,kills), onevx]


    def get_rounds(self, match_stats: MatchStats_t, history: History, ignmat: IGNMatrix) -> dict:
        """Creates a dictionary containing the sheet data for each round"""
        return {f"Round {self.__get_roundn(round_stats)}": self.__aggregate_round_data(round_stats, hround, ignmat)
                for round_stats, hround in zip(match_stats, history.get_rounds())
                if len(round_stats) > 0}

    def __get_roundn(self, round_stats: RoundStats_t) -> int:
        k0 = next(iter(round_stats.keys()))
        return round_stats[k0].rnd

    def __aggregate_round_data(self,
                               round_stats: RoundStats_t,
                               hround: HistoryRound,
                               ignmat: IGNMatrix) -> list[list[Any]]:
        return (
            ROUND_SHEET_STATS_HEADERS +
            self.compile_round_statistics(round_stats, hround, ignmat) +
                EMPTY_ROW +
            ROUND_SHEET_INFO_HEADERS +
            self.compile_round_info(hround) + 
                EMPTY_ROW +
            ROUND_SHEET_KILLFEED_HEADERS +
            self.compile_round_killfeed(hround)
        )

    def compile_round_statistics(self,
                                 round_stats: RoundStats_t,
                                 hround: HistoryRound,
                                 ignmat: IGNMatrix) -> list[list[Any]]:
        players = get_players(hround, ignmat)
        return [self.__compile_player_stats_row(round_stats[pl.uid], pl)
                for pl in players if pl.uid in round_stats]

    def __compile_player_stats_row(self, prs: PlayerRoundStats, pl: Player) -> list[Any]:
        ## "Player", "Team Index", "Kills", "Death", "Assissts", "Hs%", "Headshots", "1vX", "Operator", "Traded Kills", "Refragged Kills", "Traded Deaths"
        return [
            pl.ign,
            pl.team,
            prs.kills,
            BOOL_MAP[prs.death == 1],
            "",                             ## cannot track assists
            fmt_s(prs.headshots,prs.kills),
            prs.headshots,
            prs.onevx,
            "",                             ## TODO: track operators
            prs.traded_kills,
            prs.refrag_kills,
            prs.traded_death
        ]

    
    def compile_round_info(self, hround: HistoryRound) -> list[list[Any]]:
        okd_pign, okd_tign, okd_time = self.__get_opening_kd(hround)
        return [
            ["Site"],
            ["Winning team",  WINNING_TEAM_TEXT[hround.winner]], ## TODO: custom team names
            ["Win condition", hround.win_condition.value],
            ["Opening kill",  okd_pign, okd_time],
            ["Opening death", okd_tign, okd_time],
            ["Planted at",    str(ndefault(hround.bomb_planted_at, ""))],
            ["Defused at",    str(ndefault(hround.disabled_defuser_at, ""))],
        ]

    def __get_opening_kd(self, hround: HistoryRound) -> tuple[str, str, str]:
        opr = next((record for record in hround.killfeed if record.is_valid), None)
        if opr is None:
            return ("", "", "")
        else:
            return opr.player.ign, opr.target.ign, str(opr.time)
    
    def compile_round_killfeed(self, hround: HistoryRound) -> list[list[Any]]:
        ## "Player", "Target", "Time", "Trade", "Refrag"
        return [
            [
                record.player.ign,
                record.target.ign,
                str(record.time),
                BOOL_MAP[traded_pl is not None],
                BOOL_MAP[refrag_pl is not None]
            ]
            for record, traded_pl, refrag_pl in iter_killfeed(hround.killfeed)]

    ## Save workbook
    def create_workbook(self) -> Workbook:
        wb = Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]
        return wb
    
    @handle_write_errors
    def save_workbook(self, wb: Workbook, data: dict[str,list]) -> None:
        ## Add new sheets and fill them with data
        for sheet_name, sheet_data in data.items():
            sheet: Worksheet = wb.create_sheet(title=sheet_name)
            for row in sheet_data:
                sheet.append(row)

        wb.save(self._save_path)
