# Payoff, P&L, and Price: A Short Foundations Note

This note clarifies three terms you’ll see everywhere in the lab. Getting them right from the start avoids confusion as you add more strategies and code.

---

## Payoff vs P&L vs Price

**Payoff (at expiration)**  
The cash flow from the derivative *at expiry only*, with no regard to what you paid to enter. Example: long call payoff = max(S − K, 0). It answers: *“If the stock is S at expiry, what does the option deliver?”* Payoff diagrams in this repo show exactly that—no option premium, no cost of the stock.

**Profit/loss (P&L)**  
The gain or loss from the *entire strategy*, often including what you paid (cost basis). Example: covered call P&L = (S − S0) minus the call payoff you owe. It answers: *“How much did I make or lose on this position?”* So when a function subtracts the initial stock price S0 or adds premiums, it’s P&L, not “payoff” in the strict sense.

**Price (e.g. Black–Scholes)**  
The *value today* of the option under a model. It is a present value, not an expiration outcome. It answers: *“What should I pay *now* for this right?”* In this repo, the Black–Scholes functions return price today; the payoff scripts return what happens at expiration.

---

## Why Expiration Payoff ≠ Value Today

An option’s value today depends on time left, volatility, and interest rates—not only on where the stock might end up. A payoff diagram shows one slice: *at expiration*, for each possible stock price. So:

- **Payoff diagram** = “If we get to expiry and the stock is S, what’s the payoff (or strategy P&L)?”
- **Black–Scholes price** = “Given today’s S, time to expiry, vol, and rate, what is the option worth *right now*?”

They’re related (today’s price is a kind of “expected payoff, discounted”) but they are not the same thing.

---

## How Black–Scholes Fits In

The Black–Scholes module gives **prices today** for European options under standard assumptions (no dividends, constant rate and volatility, lognormal stock). Use it when you want “what is this option worth now?” Use the payoff (and P&L) functions when you want “what happens at expiration?” Together they give you both the *current value* and the *expiration outcome*.

---

## Why Terminology Matters

As the repo grows—more strategies, Greeks, maybe simple portfolios—mixing “payoff,” “P&L,” and “price” leads to bugs and wrong intuition. For example, comparing a payoff to a price as if they were the same, or forgetting that a “payoff” diagram might ignore premiums. We keep names and docstrings precise (e.g. “expiration payoff” vs “expiration P&L”) so that the code stays clear and the finance stays correct.
