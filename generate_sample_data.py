"""Generates SYNTHETIC lead data for testing the scoring pipeline.
BAS has no real client history yet, so this is NOT real data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 400
df = pd.DataFrame({
    "service": rng.choice(["Pulse", "Nawa", "Forward"], n, p=[.4, .3, .3]),
    "company_size": rng.choice(["1-10", "11-50", "51-200", "200+"], n, p=[.3, .35, .25, .1]),
    "source": rng.choice(["Referral", "LinkedIn", "Event", "Cold outreach"], n, p=[.25, .35, .2, .2]),
    "contacts_made": rng.integers(1, 8, n),
    "days_since_contact": rng.integers(0, 60, n),
    "meeting_held": rng.integers(0, 2, n),
})
# Assumed relationships used to simulate outcomes (assumptions, not findings)
z = (-1.5 + 1.2*(df.source == "Referral") + 0.4*(df.source == "Event")
     - 0.5*(df.source == "Cold outreach") + 1.3*df.meeting_held
     + 0.2*df.contacts_made - 0.03*df.days_since_contact
     + 0.4*(df.company_size.isin(["11-50", "51-200"])))
df["converted"] = (rng.random(n) < 1/(1+np.exp(-z))).astype(int)
df.to_csv("sample_leads.csv", index=False)
print(df.converted.mean())
