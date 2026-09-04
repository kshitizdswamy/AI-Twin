import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_activity_data(num_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic smartphone activity logs & text interactions
    over a specified number of days with weekly seasonality and autoregressive trends.
    """
    np.random.seed(seed)
    random.seed(seed)

    start_date = datetime(2025, 1, 1)
    data = []

    sample_texts_positive = [
        "Had an incredibly productive day finishing my machine learning project! Feeling proud.",
        "Great workout session this morning, feeling energized and ready to take on the week.",
        "Spent quality time with friends and had a wonderful dinner. Life is good!",
        "Crushed my targets at work today and got great feedback from the team.",
        "Learned a lot today reading papers on NLP and transformer architectures."
    ]

    sample_texts_anxious = [
        "Feeling overwhelmed by deadlines and too many unread notifications. Need a break.",
        "Stayed up way too late scrolling social media, feeling exhausted and anxious.",
        "Stressed out about upcoming exams and project presentations tomorrow.",
        "My screen time is spiraling out of control. I feel constantly distracted.",
        "Hard to focus today, too much noise and too many emails to catch up on."
    ]

    sample_texts_neutral = [
        "Routine day. Cleaned up the apartment and organized my task list.",
        "Attended meetings and worked through standard backlog items.",
        "Checked weather forecast and booked a haircut for the weekend.",
        "Commuted to office, listened to a tech podcast on the way.",
        "Prepared lunch, answered messages, and did some light reading before bed."
    ]

    sample_texts_sad = [
        "Feeling quite low today, missed spending time with family.",
        "Disappointed with my recent progress, need to reset my routine.",
        "Tired and unmotivated today. Didn't manage to accomplish much."
    ]

    prev_screen_time = 260.0

    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        day_of_week = current_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Day of week base profile (Higher usage Friday-Sunday)
        dow_factor = [230, 240, 245, 250, 280, 340, 330][day_of_week]
        
        # Autoregressive component + seasonal factor + noise
        base_screen_time = 0.6 * prev_screen_time + 0.4 * dow_factor + np.random.normal(0, 15)
        base_screen_time = max(110.0, min(600.0, base_screen_time))

        # Occasional anomaly spikes
        is_anomaly = 1 if (np.random.rand() < 0.08 or base_screen_time > 410) else 0
        if is_anomaly and np.random.rand() < 0.5:
            base_screen_time += np.random.uniform(80, 150)

        prev_screen_time = base_screen_time

        # App breakdown
        social_ratio = 0.42 if is_weekend else 0.28
        prod_ratio = 0.15 if is_weekend else 0.38
        ent_ratio = 0.28
        gaming_ratio = max(0.02, 1.0 - (social_ratio + prod_ratio + ent_ratio))

        social_mins = round(base_screen_time * social_ratio, 1)
        prod_mins = round(base_screen_time * prod_ratio, 1)
        ent_mins = round(base_screen_time * ent_ratio, 1)
        gaming_mins = round(base_screen_time * gaming_ratio, 1)

        total_screen_time = round(social_mins + prod_mins + ent_mins + gaming_mins, 1)

        # Device metrics
        notifications = int(total_screen_time * 0.3 + np.random.normal(30, 10))
        unlocks = int(total_screen_time * 0.18 + np.random.normal(15, 5))
        battery_drain_pct = min(100.0, round(total_screen_time * 0.14 + np.random.normal(15, 3), 1))

        # Pick text sample
        if total_screen_time > 380 or is_anomaly:
            text_sample = random.choice(sample_texts_anxious)
        elif prod_mins > 110:
            text_sample = random.choice(sample_texts_positive)
        elif np.random.rand() < 0.15:
            text_sample = random.choice(sample_texts_sad)
        else:
            text_sample = random.choice(sample_texts_neutral)

        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "screen_time_mins": total_screen_time,
            "social_media_mins": social_mins,
            "productivity_mins": prod_mins,
            "entertainment_mins": ent_mins,
            "gaming_mins": gaming_mins,
            "notifications": max(10, notifications),
            "unlocks": max(10, unlocks),
            "battery_drain_pct": max(10.0, battery_drain_pct),
            "is_anomaly": is_anomaly,
            "user_text": text_sample
        })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_synthetic_activity_data()
    print(f"Generated synthetic dataset: {df.shape}")
