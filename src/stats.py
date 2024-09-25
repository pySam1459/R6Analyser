from dataclasses import dataclass
from typing import Generator, Optional, TypeAlias

from history import History, HistoryRound, KFRecord, TradedRecord
from ignmatrix import Player
from utils import Timestamp
from utils.enums import Team


@dataclass
class PlayerRoundStats:
    puid: int
    rnd:  int

    kills:         int
    headshots:     int
    death:         int  # bool
    objective:     int  # bool
    traded_kills:  int  ## puid got the trading kill
    refrag_kills:  int  ## puid killed the opp who just killed
    traded_death:  int  ## puid's death was traded
    onevx:         int

    disconnected:  bool


RoundStats_t: TypeAlias = dict[int, PlayerRoundStats]
def compile_round_stats(hround: HistoryRound, players: list[Player]) -> RoundStats_t:
    round_stats = {pl.uid: PlayerRoundStats(pl.uid,
                                            hround.round_number,
                                            0, 0, 0, 0, 0, 0, 0, 0,
                                            pl in hround.disconnects)
                   for pl in players}

    __read_killfeed(round_stats, hround)
    __check_onevx(round_stats, hround, players)
    return round_stats


IterKillfeedRet_t: TypeAlias = tuple[KFRecord, Optional[Player], Optional[Player]]
def iter_killfeed(killfeed: list[KFRecord]) -> Generator[IterKillfeedRet_t, None, None]:
    seen_records = list[TradedRecord]()
    for record in killfeed:
        if not record.is_valid:
            yield (record, None, None)

        traded_pl = refrag_pl = None
        for srec in seen_records:
            if srec.time - record.time > 6 or srec.traded:
                continue
            
            if record.target.team == srec.player.team:
                traded_pl = srec.target
                srec.traded = True

            if record.target == srec.player:
                refrag_pl = srec.target
            
            if traded_pl is not None:
                break

        yield (record, traded_pl, refrag_pl)
        seen_records.append(TradedRecord(record))


def __read_killfeed(round_stats: RoundStats_t, hround: HistoryRound) -> None:
    for record, traded_pl, refrag_pl in iter_killfeed(hround.killfeed):
        round_stats[record.player.uid].kills += 1
        round_stats[record.player.uid].headshots += record.headshot
        round_stats[record.target.uid].death = 1

        if traded_pl is not None:
            round_stats[record.player.uid].traded_kills += 1
            round_stats[traded_pl.uid].traded_death = 1
        if refrag_pl is not None:
            round_stats[record.player.uid].refrag_kills += 1


@dataclass
class _Death:
    player: Player
    time: Timestamp


def __check_onevx(round_stats: RoundStats_t, hround: HistoryRound, players: list[Player]) -> None:
    team0 = [pl for pl in players if pl.team == Team.TEAM0]
    team1 = [pl for pl in players if pl.team == Team.TEAM1]

    kf_deaths = [_Death(record.target, record.time) for record in hround.killfeed]
    dc_deaths = [_Death(dc.player, dc.time) for dc in hround.disconnects]
    all_deaths = sorted(kf_deaths + dc_deaths, key=lambda d: d.time, reverse=True) ## start with first death

    for d in all_deaths + [None]:
        if (len(team0) == 1 and hround.winner == Team.TEAM0 and len(team1) >= 1):
            round_stats[team0[0].uid].onevx = len(team1)
            return
        elif (len(team1) == 1 and hround.winner == Team.TEAM1 and len(team0) >= 1):
            round_stats[team1[0].uid].onevx = len(team0)
            return

        if d is None:
            return
        if d.player.team == Team.TEAM0:
            team0 = [pl for pl in team0 if pl.uid != d.player.uid]
        elif d.player.team == Team.TEAM1:
            team1 = [pl for pl in team1 if pl.uid != d.player.uid]


MatchStats_t: TypeAlias = list[RoundStats_t]
def compile_match_stats(history: History, players: list[Player]) -> MatchStats_t:
    return [compile_round_stats(hround, players) for hround in history.get_rounds()]


def is_kost(prs: PlayerRoundStats) -> int:
    return prs.kills > 0 or prs.death == 0 or prs.objective == 1 or prs.traded_kills > 0