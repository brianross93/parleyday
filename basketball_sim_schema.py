from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BasketballSide(StrEnum):
    OFFENSE = "offense"
    DEFENSE = "defense"


class CourtZone(StrEnum):
    BACKCOURT = "backcourt"
    TOP = "top"
    LEFT_WING = "left_wing"
    RIGHT_WING = "right_wing"
    LEFT_CORNER = "left_corner"
    RIGHT_CORNER = "right_corner"
    LEFT_DUNKER = "left_dunker"
    RIGHT_DUNKER = "right_dunker"
    NAIL = "nail"
    ELBOW_LEFT = "elbow_left"
    ELBOW_RIGHT = "elbow_right"
    PAINT = "paint"
    RIM = "rim"


class OffensiveRole(StrEnum):
    PRIMARY_CREATOR = "primary_creator"
    SECONDARY_CREATOR = "secondary_creator"
    MOVEMENT_SHOOTER = "movement_shooter"
    SPACER = "spacer"
    ROLL_BIG = "roll_big"
    POP_BIG = "pop_big"
    POST_HUB = "post_hub"
    SLASHER = "slasher"
    GLUE = "glue"


class DefensiveRole(StrEnum):
    POINT_OF_ATTACK = "point_of_attack"
    WING_STOPPER = "wing_stopper"
    HELPER = "helper"
    RIM_PROTECTOR = "rim_protector"
    REBOUNDER = "rebounder"
    SWITCH_BIG = "switch_big"


class PlayFamily(StrEnum):
    TRANSITION_PUSH = "transition_push"
    EARLY_DRAG = "early_drag"
    HIGH_PICK_AND_ROLL = "high_pick_and_roll"
    DOUBLE_DRAG = "double_drag"
    HORNS = "horns"
    FIVE_OUT = "five_out"
    CHICAGO_ACTION = "chicago_action"
    SPAIN_PICK_AND_ROLL = "spain_pick_and_roll"
    POST_TOUCH = "post_touch"
    STAGGER = "stagger"
    ISO = "iso"
    HANDOFF = "handoff"
    RESET = "reset"


class DefensiveCoverage(StrEnum):
    MAN = "man"
    SWITCH = "switch"
    DROP = "drop"
    HEDGE = "hedge"
    BLITZ = "blitz"
    ICE = "ice"
    ZONE = "zone"
    SCRAM = "scram"


class PossessionPhase(StrEnum):
    ADVANCE = "advance"
    INITIATION = "initiation"
    PRIMARY_ACTION = "primary_action"
    SECONDARY_ACTION = "secondary_action"
    SHOT = "shot"
    REBOUND = "rebound"
    DEAD_BALL = "dead_ball"


class EventType(StrEnum):
    INBOUND = "inbound"
    ADVANCE = "advance"
    PASS = "pass"
    SCREEN = "screen"
    HANDOFF = "handoff"
    CUT = "cut"
    DRIVE = "drive"
    POST_ENTRY = "post_entry"
    SHOT = "shot"
    FOUL = "foul"
    TURNOVER = "turnover"
    REBOUND = "rebound"
    SUBSTITUTION = "substitution"
    TIMEOUT = "timeout"
    VIOLATION = "violation"


class ShotType(StrEnum):
    RIM = "rim"
    PAINT = "paint"
    MIDRANGE = "midrange"
    CORNER_THREE = "corner_three"
    ABOVE_BREAK_THREE = "above_break_three"
    FREE_THROW = "free_throw"


class TurnoverType(StrEnum):
    BAD_PASS = "bad_pass"
    STRIP = "strip"
    TRAVEL = "travel"
    CHARGE = "charge"
    SHOT_CLOCK = "shot_clock"
    OFFENSIVE_FOUL = "offensive_foul"


class FoulOutcomeType(StrEnum):
    AND_ONE = "and_one"
    TWO_SHOT = "two_shot"
    THREE_SHOT = "three_shot"


@dataclass(frozen=True)
class CourtPoint:
    x: float
    y: float
    zone: CourtZone


@dataclass(frozen=True)
class PlayerTraitProfile:
    ball_security: float
    separation: float
    burst: float
    pullup_shooting: float
    catch_shoot: float
    finishing: float
    pass_vision: float
    pass_accuracy: float
    decision_making: float
    screen_setting: float
    oreb: float
    free_throw_rating: float
    ft_pct_raw: float
    foul_drawing: float
    containment: float
    closeout: float
    screen_nav: float
    interior_def: float
    rim_protect: float
    steal_pressure: float
    dreb: float
    foul_discipline: float
    help_rotation: float
    stamina: float
    role_consistency: float = 10.0
    clutch: float = 0.0
    size: float = 10.0
    reach: float = 10.0

    @property
    def catch_and_shoot(self) -> float:
        return self.catch_shoot

    @property
    def offensive_rebounding(self) -> float:
        return self.oreb

    @property
    def defensive_rebounding(self) -> float:
        return self.dreb

    @property
    def perimeter_defense(self) -> float:
        return self.containment

    @property
    def interior_defense(self) -> float:
        return self.interior_def

    @property
    def rim_protection(self) -> float:
        return self.rim_protect

    @property
    def passing_creation(self) -> float:
        return (self.pass_vision + self.pass_accuracy + self.decision_making) / 3.0

    @property
    def turnover_tendency(self) -> float:
        return max(1.0, 21.0 - ((self.ball_security * 0.6) + (self.decision_making * 0.4)))

    @property
    def usage(self) -> float:
        return self.offensive_load

    @property
    def offensive_load(self) -> float:
        return (
            (0.35 * self.separation)
            + (0.25 * self.pullup_shooting)
            + (0.15 * self.ball_security)
            + (0.15 * self.finishing)
            + (0.10 * self.foul_drawing)
        )


@dataclass(frozen=True)
class PlayerCondition:
    energy: float = 1.0
    fatigue: float = 0.0
    foul_count: int = 0
    confidence: float = 0.5
    minutes_played: float = 0.0
    available: bool = True


@dataclass(frozen=True)
class PlayerSimProfile:
    player_id: str
    name: str
    team_code: str
    positions: tuple[str, ...]
    offensive_role: OffensiveRole
    defensive_role: DefensiveRole
    traits: PlayerTraitProfile
    condition: PlayerCondition = field(default_factory=PlayerCondition)


@dataclass(frozen=True)
class TeamTactics:
    pace_target: float
    transition_frequency: float
    crash_glass_rate: float
    help_aggressiveness: float
    switch_rate: float
    zone_rate: float
    no_middle_rate: float
    pre_switch_rate: float
    rotation_tightness: float
    late_clock_isolation_rate: float
    early_offense_rate: float
    pick_and_roll_rate: float
    handoff_rate: float
    post_touch_rate: float
    off_ball_screen_rate: float
    play_family_weights: dict[PlayFamily, float]
    coverage_weights: dict[DefensiveCoverage, float]
    star_usage_bias: float = 1.0
    closeout_attack_rate: float = 0.5
    second_side_rate: float = 0.25
    corner_spacing_bias: float = 0.5
    shooter_distribution_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FloorPlayerState:
    player_id: str
    side: BasketballSide
    location: CourtPoint
    has_ball: bool = False


@dataclass(frozen=True)
class LineupUnit:
    team_code: str
    player_ids: tuple[str, ...]
    spacing_score: float
    creation_score: float
    rim_pressure_score: float
    rebounding_score: float
    switchability_score: float
    rim_protection_score: float


@dataclass(frozen=True)
class DefensiveAssignment:
    defender_id: str
    offensive_player_id: str
    matchup_strength: float
    on_ball: bool = False
    help_priority: float = 0.0


@dataclass(frozen=True)
class GameClockState:
    period: int
    seconds_remaining_in_period: float
    shot_clock: float
    possession_number: int


@dataclass(frozen=True)
class ScoreState:
    offense_score: int
    defense_score: int
    bonus_offense: bool = False
    bonus_defense: bool = False


@dataclass(frozen=True)
class PlayCall:
    family: PlayFamily
    primary_actor_id: str
    secondary_actor_id: str | None = None
    screener_id: str | None = None
    target_zone: CourtZone | None = None
    intended_shot_type: ShotType | None = None
    tempo_pressure: float = 0.0


@dataclass(frozen=True)
class PossessionContext:
    offense_team_code: str
    defense_team_code: str
    clock: GameClockState
    score: ScoreState
    offense_lineup: LineupUnit
    defense_lineup: LineupUnit
    offensive_tactics: TeamTactics
    defensive_tactics: TeamTactics
    floor_players: tuple[FloorPlayerState, ...]
    defensive_assignments: tuple[DefensiveAssignment, ...]
    player_pool: tuple[PlayerSimProfile, ...]
    current_phase: PossessionPhase
    play_call: PlayCall | None = None
    coverage: DefensiveCoverage | None = None


@dataclass(frozen=True)
class EventContext:
    event_type: EventType
    actor_id: str | None = None
    receiver_id: str | None = None
    defender_id: str | None = None
    location: CourtPoint | None = None
    shot_type: ShotType | None = None
    turnover_type: TurnoverType | None = None
    success_probability: float = 0.0
    realized_success: bool | None = None
    points_scored: int = 0
    foul_drawn: bool = False
    notes: str = ""


@dataclass(frozen=True)
class PossessionOutcome:
    points_scored: int
    made_shot: bool
    turnover: bool
    foul_committed: bool
    offensive_rebound: bool
    shooting_player_id: str | None
    assisting_player_id: str | None
    rebounder_id: str | None
    turnover_player_id: str | None
    events: tuple[EventContext, ...]
    free_throws_attempted: int = 0
    foul_type: FoulOutcomeType | None = None
    steal_player_id: str | None = None
    block_player_id: str | None = None


@dataclass(frozen=True)
class RotationPlan:
    starters: tuple[str, ...]
    closing_group: tuple[str, ...]
    stagger_pairs: tuple[tuple[str, str], ...] = ()
    max_stint_minutes: dict[str, float] = field(default_factory=dict)
    target_minutes: dict[str, float] = field(default_factory=dict)
    backup_priority: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class GameSimulationInput:
    game_id: str
    home_team_code: str
    away_team_code: str
    players: tuple[PlayerSimProfile, ...]
    home_tactics: TeamTactics
    away_tactics: TeamTactics
    home_rotation: RotationPlan
    away_rotation: RotationPlan
    opening_tip_winner: str | None = None


@dataclass(frozen=True)
class TeamBoxScoreProjection:
    team_code: str
    points: float
    field_goal_attempts: float
    threes_attempted: float
    free_throws_attempted: float
    turnovers: float
    offensive_rebounds: float
    defensive_rebounds: float
    assists: float


@dataclass(frozen=True)
class PlayerBoxScoreProjection:
    player_id: str
    name: str
    minutes: float
    points: float
    rebounds: float
    assists: float
    steals: float
    blocks: float
    turnovers: float
    fouls: float


@dataclass(frozen=True)
class GameSimulationResult:
    game_id: str
    home_team_code: str
    away_team_code: str
    home_score: int
    away_score: int
    possession_count: int
    event_log: tuple[EventContext, ...]
    player_box_scores: tuple[PlayerBoxScoreProjection, ...]
    team_box_scores: tuple[TeamBoxScoreProjection, ...]
