"""The public release contains only the long twelve-rover demonstrations."""
import json
from pathlib import Path

def test_public_media_has_only_twelve_rover_gifs_and_videos():
    media=Path('media')
    expected={'rover_recovery_12','rover_reallocation_vs_fixed_12'}
    for ext in ('gif','mp4'):
        assert {p.stem for p in media.glob(f'*.{ext}')} == expected
    assert not (media/'rover_recovery_trace.json').exists()
    for stem in expected:
        obj=json.loads((media/(stem+'_trace.json')).read_text())
        for side in ('left','right'):
            states=obj[side]['states']
            assert len(states) == 121
            assert len(states[0]['active']) == 14
            assert sum(states[0]['active']) == 12
            assert len(states[0]['served']) == 6

def test_default_demo_uses_twelve_rover_actor_and_ignored_outputs():
    script=Path('scripts/make_demo.py').read_text()
    for required in (
        "default='examples/checkpoints/team12_seed_314159.pt'",
        "default='configs/env/rover_team_12_showcase.yaml'",
        "default='configs/env/rover_team_12.yaml'",
        "default='outputs/demo_12_vs_fixed.gif'",
        "default='outputs/demo_12_vs_fixed.mp4'",
        "default='fixed'",
    ):
        assert required in script
