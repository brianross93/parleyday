Original prompt: okay in the meantime lets work on the visual rendering side and shot location and how distance affects shooting and such

- 2026-04-03: Starting spatial shot-quality pass.
- Goal for this chunk: add shared court geometry, richer shot coordinates, and distance-aware shot quality that both the sim and renderer can use.
- Follow-up goal: inspect match view after backend changes so the new locations actually look plausible on court.
- 2026-04-03: Added shared court geometry in basketball_court.py with real NBA dimensions and distance-aware shot context.
- 2026-04-03: Shot events now carry location, shot distance, defender distance proxy, and zone labels through the possession engine and viewer payload.
- 2026-04-03: Match-view broadcaster copy now includes shot distance on jumpers.
- TODO: Add renderer-side overlays for shot zones/hexes and improve spatial continuity for non-shot events using the same court geometry layer.
- 2026-04-03: Movement templates now use the shared court geometry anchors for base spacing and clamps.
- 2026-04-03: Shot resolver now blends sampled shot points with action-origin points so side actions and kickouts produce location-aware jumpers.
- 2026-04-03: Match view now renders subtle zone overlays and a live shot marker/tag with zone + distance.
- TODO: Feed defender positions directly into shot-origin and contest selection instead of only using trait-based contest proxies.

- 2026-04-03: shot resolver now blends real defender geometry into contest distance, transition trail-threes use live floor positions, and final shot type follows the resolved shot point so inside-arc locations no longer stay mislabeled as threes.
- 2026-04-04: choreography now caps per-beat actor travel, lengthens segments based on movement distance, and softens off-ball/help blend weights so players stop sliding across huge distances in a single beat.
- 2026-04-04: possession changes now use a truer reset sequence: made-basket possessions begin from an inbound layout, while outlet/turnover possessions start from a backcourt push instead of a full-court teleport.
- 2026-04-04: route smoke and Playwright browser setup completed; latest screenshot artifact was captured at `/Users/brianross/Desktop/parleyday/output/web-game/shot-0.png`.
- 2026-04-04: added an explicit `Inbound Pass` beat between made-basket possession change and the walk-up, so inbounded possessions do not jump directly into the push.
- 2026-04-04: weak-side offense and defense now use authored activity targets during `screen`, `handoff`, and `drive` beats instead of relocating to one static shell.
- 2026-04-04: actor tracks now carry trait-driven motion metadata (`delay`, `tempo`) derived from `speed`, `burst`, `stamina`, `containment`, `screen_nav`, `closeout`, and `ball_handle`, and the viewer interpolation uses those fields so players arrive on the same beat at meaningfully different rates.
- TODO: The next visual realism pass should add authored jogging/backpedal/inbound animations or multi-segment possession-start sequences; the current fix improves continuity but does not yet make weak-side defenders feel fully “alive.”
