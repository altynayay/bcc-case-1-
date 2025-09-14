import pandas as pd
import numpy as np
import glob
import json
from datetime import datetime

def analyze_all_clients():
    """Анализирует всех клиентов и генерирует рекомендации"""
    
    # Загружаем профили клиентов
    clients_df = pd.read_csv('clients.csv')
    
    # Загружаем все транзакции
    transaction_files = glob.glob('client_*_transactions_3m.csv')
    all_transactions = []
    for file in transaction_files:
        df = pd.read_csv(file)
        all_transactions.append(df)
    transactions_df = pd.concat(all_transactions, ignore_index=True)
    transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Загружаем все переводы
    transfer_files = glob.glob('client_*_transfers_3m.csv')
    all_transfers = []
    for file in transfer_files:
        df = pd.read_csv(file)
        all_transfers.append(df)
    transfers_df = pd.concat(all_transfers, ignore_index=True)
    transfers_df['date'] = pd.to_datetime(transfers_df['date'])
    
    recommendations = []
    
    for _, client in clients_df.iterrows():
        client_code = client['client_code']
        
        # Получаем транзакции клиента
        client_transactions = transactions_df[transactions_df['client_code'] == client_code]
        client_transfers = transfers_df[transfers_df['client_code'] == client_code]
        
        # Анализируем траты по категориям
        category_spending = client_transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Анализируем переводы
        transfer_analysis = client_transfers.groupby('type')['amount'].sum()
        
        # Рассчитываем выгоду для каждого продукта
        benefits = calculate_benefits(client, category_spending, transfer_analysis, client_transactions)
        
        # Выбираем лучший продукт
        if benefits:
            best_product = max(benefits.keys(), key=lambda x: benefits[x]['benefit'])
            best_benefit = benefits[best_product]
            
            # Генерируем пуш-уведомление
            push_message = generate_push_message(client['name'], best_product, category_spending, best_benefit)
            
            recommendations.append({
                'client_code': client_code,
                'name': client['name'],
                'status': client['status'],
                'age': client['age'],
                'city': client['city'],
                'balance': client['avg_monthly_balance_KZT'],
                'product': best_product,
                'benefit': best_benefit['benefit'],
                'reason': best_benefit['reason'],
                'push_notification': push_message
            })
    
    return recommendations

def calculate_benefits(client, category_spending, transfer_analysis, client_transactions):
    """Рассчитывает выгоду для каждого продукта"""
    benefits = {}
    
    # Карта для путешествий
    travel_spending = category_spending.get('Путешествия', 0) + category_spending.get('Такси', 0) + category_spending.get('Отели', 0)
    travel_benefit = travel_spending * 0.04
    if travel_benefit > 0:
        benefits['Карта для путешествий'] = {
            'benefit': travel_benefit,
            'reason': f"Кешбэк 4% на путешествия: {travel_benefit:,.0f} ₸"
        }
    
    # Премиальная карта
    total_spending = client_transactions['amount'].sum()
    premium_spending = category_spending.get('Кафе и рестораны', 0) + category_spending.get('Косметика и Парфюмерия', 0) + category_spending.get('Ювелирные украшения', 0)
    
    base_benefit = total_spending * 0.02
    premium_benefit = premium_spending * 0.04
    
    if client['avg_monthly_balance_KZT'] > 1000000:
        tier_multiplier = 1.5
    else:
        tier_multiplier = 1.0
        
    premium_total = (base_benefit + premium_benefit) * tier_multiplier
    if premium_total > 0:
        benefits['Премиальная карта'] = {
            'benefit': premium_total,
            'reason': f"Кешбэк {2*tier_multiplier}% на все + 4% на люкс: {premium_total:,.0f} ₸"
        }
    
    # Кредитная карта
    top_categories = list(category_spending.keys())[:3]
    favorite_benefit = sum([category_spending.get(cat, 0) * 0.10 for cat in top_categories])
    online_benefit = (category_spending.get('Едим дома', 0) + category_spending.get('Смотрим дома', 0) + category_spending.get('Играем дома', 0)) * 0.10
    credit_total = favorite_benefit + online_benefit
    if credit_total > 0:
        benefits['Кредитная карта'] = {
            'benefit': credit_total,
            'reason': f"10% в топ-категориях + онлайн: {credit_total:,.0f} ₸"
        }
    
    # Обмен валют
    fx_volume = transfer_analysis.get('fx_buy', 0) + transfer_analysis.get('fx_sell', 0)
    fx_benefit = fx_volume * 0.005
    if fx_benefit > 0:
        benefits['Обмен валют'] = {
            'benefit': fx_benefit,
            'reason': f"Экономия на обмене: {fx_benefit:,.0f} ₸"
        }
    
    # Депозиты
    balance = client['avg_monthly_balance_KZT']
    if balance > 500000:
        deposit_benefit = balance * 0.12 / 12
        benefits['Депозит сберегательный'] = {
            'benefit': deposit_benefit,
            'reason': f"Доход с депозита: {deposit_benefit:,.0f} ₸/мес"
        }
    
    # Инвестиции
    if balance > 100000:
        invest_benefit = balance * 0.08 / 12
        benefits['Инвестиции'] = {
            'benefit': invest_benefit,
            'reason': f"Потенциальный доход: {invest_benefit:,.0f} ₸/мес"
        }
    
    return benefits

def generate_push_message(name, product, category_spending, benefit_info):
    """Генерирует персонализированное пуш-уведомление"""
    templates = {
        "Карта для путешествий": f"{name}, в последние месяцы вы много путешествуете и пользуетесь такси. С картой для путешествий вернули бы {benefit_info['benefit']:,.0f} ₸ кешбэка. Откройте карту.",
        "Премиальная карта": f"{name}, у вас стабильно крупный остаток и активные траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформите сейчас.",
        "Кредитная карта": f"{name}, ваши топ-категории — {', '.join(list(category_spending.keys())[:3])}. Кредитная карта даёт до 10% в любимых категориях. Оформите карту.",
        "Обмен валют": f"{name}, вы часто обмениваете валюту. В приложении выгодный обмен и авто-покупка по целевому курсу. Настройте обмен.",
        "Депозит сберегательный": f"{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Откройте вклад.",
        "Инвестиции": f"{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Откройте счёт."
    }
    
    return templates.get(product, f"{name}, у нас есть выгодное предложение для вас. Узнайте подробнее.")

if __name__ == "__main__":
    print("Анализируем всех 60 клиентов...")
    recommendations = analyze_all_clients()
    
    # Сохраняем в JSON
    with open('all_recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, ensure_ascii=False, indent=2)
    
    # Сохраняем в CSV
    df = pd.DataFrame(recommendations)
    df.to_csv('all_recommendations.csv', index=False)
    
    print(f"✅ Анализ завершен! Создано {len(recommendations)} рекомендаций")
    print("📁 Файлы сохранены: all_recommendations.json, all_recommendations.csv")
