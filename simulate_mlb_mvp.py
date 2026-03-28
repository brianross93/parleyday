import json

from monte_carlo.mlb import MLBGameConfig, MLBGameSimulator, build_demo_mlb_matchup


def main() -> None:
    away, home, props = build_demo_mlb_matchup()
    simulator = MLBGameSimulator(MLBGameConfig(n_simulations=2000, random_seed=7))
    result = simulator.simulate_game(away=away, home=home, market_props=props)
    edges = simulator.evaluate_edges(result, props)

    payload = {
        "away_team": result.away_team,
        "home_team": result.home_team,
        "average_score": {
            result.away_team: sum(result.away_scores) / len(result.away_scores),
            result.home_team: sum(result.home_scores) / len(result.home_scores),
        },
        "edges": [
            {
                "player": edge.player_name,
                "stat": edge.stat,
                "line": edge.line,
                "side": edge.side,
                "sim_probability": round(edge.sim_probability, 4),
                "market_price": round(edge.market_price, 4),
                "edge": round(edge.edge, 4),
                "confidence": round(edge.confidence, 4),
                "category": edge.category,
            }
            for edge in edges
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
