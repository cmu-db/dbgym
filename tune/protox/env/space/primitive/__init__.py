from enum import Enum, unique


@unique
class SettingType(Enum):
    INVALID = -1
    BOOLEAN = 0
    INTEGER = 1
    BYTES = 2
    INTEGER_TIME = 3
    FLOAT = 4

    BINARY_ENUM = 5
    SCANMETHOD_ENUM = 6
    SCANMETHOD_ENUM_CATEGORICAL = 7
    QUERY_TABLE_ENUM = 9


@unique
class KnobClass(Enum):
    INVALID = -1
    KNOB = 0
    TABLE = 1
    QUERY = 2


def is_knob_enum(knob_type: SettingType) -> bool:
    return knob_type in [
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM,
        SettingType.QUERY_TABLE_ENUM,
    ]


def is_boolean(knob_type: SettingType) -> bool:
    return knob_type in [
        SettingType.BOOLEAN,
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM,
    ]


def is_binary_enum(knob_type: SettingType) -> bool:
    return knob_type in [
        SettingType.BINARY_ENUM,
        SettingType.SCANMETHOD_ENUM,
    ]
