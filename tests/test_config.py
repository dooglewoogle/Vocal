"""Config loading: nested layout, type strictness, legacy-layout guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal.config import ConfigError, VocalConfig, load_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


def test_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg == VocalConfig()


def test_nested_tables_apply(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, """
        log_level = "DEBUG"

        [input.model]
        size = "medium.en"

        [input.hotkey]
        key = "F12"
        duck = true

        [input.inject]
        method = "xdotool"

        [output.speech]
        voice = "kokoro-af_sarah"
        speed = 1.2
        pause_input = false

        [output.server]
        enabled = false
        port = 5000
    """))
    assert cfg.log_level == "DEBUG"
    assert cfg.input.model.size == "medium.en"
    assert cfg.input.hotkey.key == "F12"
    assert cfg.input.hotkey.duck is True
    assert cfg.input.inject.method == "xdotool"
    assert cfg.output.speech.voice == "kokoro-af_sarah"
    assert cfg.output.speech.speed == 1.2
    assert cfg.output.speech.pause_input is False
    assert cfg.output.server.enabled is False
    assert cfg.output.server.port == 5000


def test_int_accepted_for_float_field(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "[output.speech]\nspeed = 2\n"))
    assert cfg.output.speech.speed == 2.0
    assert isinstance(cfg.output.speech.speed, float)


def test_bool_rejected_for_int_field(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duck_amount"):
        load_config(_write(tmp_path, "[input.hotkey]\nduck_amount = true\n"))


def test_wrong_type_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="port"):
        load_config(_write(tmp_path, "[output.server]\nport = \"abc\"\n"))


def test_unknown_keys_ignored(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "[input.model]\nsize = \"tiny.en\"\nbogus = 1\n\n[whatever]\nx = 1\n"))
    assert cfg.input.model.size == "tiny.en"


@pytest.mark.parametrize(
    "body, old, new",
    [
        ("[model]\nsize = 'tiny.en'\n", "[model]", "[input.model]"),
        ("[hotkey]\nkey = 'F1'\n", "[hotkey]", "[input.hotkey]"),
        ("[output]\nmethod = 'xdotool'\n", "[output]", "[input.inject]"),
        ("[postprocess]\ncapitalize_first = false\n", "[postprocess]", "[input.postprocess]"),
    ],
)
def test_legacy_layout_rejected_with_mapping(tmp_path: Path, body: str, old: str, new: str) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    msg = str(exc.value)
    assert old in msg and new in msg


def test_legacy_error_lists_all_offending_tables(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, "[model]\nsize='a'\n[audio]\nsample_rate=1\n"))
    msg = str(exc.value)
    assert "[model]" in msg and "[audio]" in msg


def test_new_output_table_not_mistaken_for_legacy(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "[output.speech]\nvoice = 'system'\n"))
    assert cfg.output.speech.voice == "system"


# ── engine key ──


def test_engine_key_loads_and_validates(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "[input]\nengine = 'hotkey'\n"))
    assert cfg.input.engine == "hotkey"
    with pytest.raises(ConfigError, match="input.engine"):
        load_config(_write(tmp_path, "[input]\nengine = 'telepathy'\n"))


def test_phrasebook_table(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, "[input.phrasebook]\nseed = true\nreplace = true\n"))
    assert cfg.input.phrasebook.seed is True and cfg.input.phrasebook.replace is True


# ── save / copy ──


def test_save_config_round_trips(tmp_path: Path) -> None:
    from vocal.config import save_config

    cfg = VocalConfig()
    cfg.input.engine = "hotkey"
    cfg.input.model.size = "tiny.en"
    cfg.input.hotkey.duck = True
    cfg.input.audio.device = "USB Mic"
    cfg.output.speech.speed = 1.3
    cfg.output.speech.voice = "kokoro-af_sarah"
    cfg.output.server.port = 5005
    cfg.log_level = "DEBUG"

    path = save_config(cfg, tmp_path / "sub" / "config.toml")
    assert path.exists() and not path.with_suffix(".toml.tmp").exists()
    assert load_config(path) == cfg


def test_save_config_omits_none_fields(tmp_path: Path) -> None:
    from vocal.config import save_config

    body = save_config(VocalConfig(), tmp_path / "c.toml").read_text()
    assert "device" not in body and "model_path" not in body
    assert "[input.model]" in body and "[output.speech]" in body


def test_copy_into_updates_nested_in_place() -> None:
    from vocal.config import copy_into

    live = VocalConfig()
    speech_ref = live.output.speech  # something else holds this reference
    new = VocalConfig()
    new.output.speech.speed = 2.0
    new.input.model.size = "medium.en"
    new.input.audio.device = "7"

    copy_into(live, new)
    assert live == new
    assert live.output.speech is speech_ref and speech_ref.speed == 2.0
    assert live.input.audio.device == "7"
