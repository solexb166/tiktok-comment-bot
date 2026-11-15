# Exness specification verification register

Each Environment-B field must be `verified` here (with source and date) before B results count.
Current machine-readable values and statuses live in `configs/brokers/exness/`. Environment B
loading fails while any required field is not verified (`UnverifiedSpecError`).

| ID | Field | Current status | Value / candidates | Source to check | Verified on |
|---|---|---|---|---|---|
| U1 | Legal entity -> region file | unknown | global / ke / jo | your Personal Area / client agreement | |
| U2 | BTCUSDc availability, contract | conflicting | excluded from B | Cryptocurrencies + Standard Cent pages | |
| U3 | USTECc availability, contract | conflicting | excluded from B | Indices + Standard Cent pages | |
| U4 | XAUUSDc contract size | unverified | 1 oz per Cent lot (derived) | Commodities page / contract spec | |
| U5 | Unlimited eligibility, margin formula | conflicting | equity < $5,000 or ~$1,000; history? | Leverage page | |
| U6 | HMR windows and bands; news calendar | conflicting | 90 s vs 5 min after news | HMR page | |
| U7 | Stop-out closing order | unverified | most unprofitable first | Margin call / stop out pages | |
| U8 | Swap values; swap-free eligibility | unknown | majors swap-free? | About swap; Forex page | |
| U9 | Cent symbols in tick history | unknown | | Tick history page | |
| U10 | Rollover time; server time zone | verified (time) | 21:00/22:00 UTC | About swap | 2026-11-15 (search extract) |
| U11 | Sessions and daily breaks | unknown | | instrument pages; tick gaps | |
| U14 | Insufficient-margin behaviour | unverified | reject (MT5 standard) | support | |
| U15 | Equity-tier change on open positions | unverified | recomputed continuously | Leverage page | |
| U16 | Fixed margins BTC / NAS100 (A placeholders) | assumption | 0.25% / 0.25% | Crypto / Indices pages | |
| - | Lot step 0.01 for Cent FX/gold | unverified | 0.01 | contract spec | |
