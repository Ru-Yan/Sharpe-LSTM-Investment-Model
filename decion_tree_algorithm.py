def all_theaten_index(sigma,alrfa_1,alrfa_2,avg_h,avg_i):
    kamma = float(sigma) / ((alrfa_1**avg_h)*(alrfa_2**avg_i)) 
    return kamma

def iof_trade(gold_data,bitcoin_data,gold_bais,
              bitcoin_bais,current_day,iof_break):
    current_gold_bais = gold_bais[current_day]
    current_bitcoin_bais = bitcoin_bais[current_day]
    theaten_index = all_theaten_index()
    if(current_bitcoin_bais >= 0.05):
        ''''''
    elif(current_bitcoin_bais <= -0.05):
        ''''''
    
    