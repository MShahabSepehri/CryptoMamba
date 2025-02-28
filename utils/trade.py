
def buy_sell_smart(today, pred, balance, shares, risk=5):
    global total_trades

    diff = pred * risk / 100
    if today > pred + diff:
        total_trades += 1 if shares > 0 else 0
        balance += shares * today
        shares = 0
    elif today > pred:
        total_trades += 1 if shares > 0 else 0
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        total_trades += 1 if balance > 0 else 0
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        total_trades += 1 if balance > 0 else 0
        shares += balance / today
        balance = 0
    return balance, shares

total_trades = 0


def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    diff = pred * risk / 100
    if today < pred - diff:
        total_trades += 1 if shares > 0 else 0
        shares += balance / today
        balance = 0
    elif today < pred:
        total_trades += 1 if balance > 0 else 0
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    elif today < pred + diff:
        if shares > 0:
            total_trades += 1 if shares > 0 else 0
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= (1 - factor)
    else:
        total_trades += 1 if balance > 0 else 0
        balance += (shares + max_n_btc) * today
        shares = -max_n_btc
    return balance, shares

def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    global total_trades
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        total_trades += 1 if balance > 0 else 0
        shares += balance / today
        balance = 0
    else:
        total_trades += 1 if shares > 0 else 0
        balance += shares * today
        shares = 0
    return balance, shares


def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close'):
    balance_in_time = [balance]
    shares = 0
    global total_trades
    total_trades = 0

    for ts, target, pred in zip(timstamps, targets, preds):
        today = data[data[time_key] == int(ts - 24 * 60 * 60)].iloc[0][y_key]
        assert round(target, 2) == round(data[data[time_key] == int(ts)].iloc[0][y_key], 2)
        if mode == 'smart':
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        if mode == 'smart_w_short':
            balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_n_btc=0.002)
        elif mode == 'vanilla':
            balance, shares = buy_sell_vanilla(today, pred, balance, shares)
        elif mode == 'no_strategy':
            shares += balance / today
            balance = 0
        balance_in_time.append(shares * today + balance)

    balance += shares * targets[-1]
    return balance, balance_in_time, total_trades