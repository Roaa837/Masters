def select_shots_knapsack(scored_shots, max_duration=5.0):

    shots = sorted(
        scored_shots,
        key=lambda x: x["score"] / max(x["data"]["end"] - x["data"]["start"], 0.01),
        reverse=True
    )

    selected = []
    total_time = 0.0

    for shot in shots:

        duration = max(shot["data"]["end"] - shot["data"]["start"], 0.01)

        if total_time + duration <= max_duration:
            selected.append(shot)
            total_time += duration

    # keep original video order
    selected.sort(key=lambda x: x["data"]["start"])

    return selected