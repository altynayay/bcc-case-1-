from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import json
from ml_analyzer import MLBankAnalyzer

app = Flask(__name__)

class ClientAnalyzer:
    def __init__(self):
        self.clients_df = None
        self.products = {
            "Карта для путешествий": {
                "categories": ["Путешествия", "Такси", "Отели"],
                "cashback_rate": 0.04,
                "description": "Повышенный кешбэк на путешествия и такси"
            },
            "Премиальная карта": {
                "categories": ["Кафе и рестораны", "Косметика и Парфюмерия", "Ювелирные украшения"],
                "base_cashback": 0.02,
                "premium_cashback": 0.04,
                "description": "Базовый кешбэк 2-4% + повышенный на рестораны и люкс"
            },
            "Кредитная карта": {
                "categories": ["Едим дома", "Смотрим дома", "Играем дома"],
                "online_cashback": 0.10,
                "favorite_cashback": 0.10,
                "description": "До 10% в любимых категориях + онлайн-сервисы"
            },
            "Обмен валют": {
                "signals": ["fx_buy", "fx_sell"],
                "description": "Выгодный обмен валют и авто-покупка"
            },
            "Кредит наличными": {
                "signals": ["loan_payment_out"],
                "description": "Быстрый доступ к финансированию"
            },
            "Депозит мультивалютный": {
                "signals": ["fx_buy", "fx_sell"],
                "description": "Проценты + удобство хранения валют"
            },
            "Депозит сберегательный": {
                "signals": ["deposit_topup_out"],
                "description": "Максимальная ставка за счет заморозки"
            },
            "Депозит накопительный": {
                "signals": ["deposit_topup_out"],
                "description": "Повышенная ставка с пополнениями"
            },
            "Инвестиции": {
                "signals": ["invest_out", "invest_in"],
                "description": "Нулевые комиссии, низкий порог входа"
            },
            "Золотые слитки": {
                "signals": ["gold_buy_out", "gold_sell_in"],
                "description": "Защитный актив и диверсификация"
            }
        }
    
    def load_data(self):
        """Загружает данные клиентов, транзакций и переводов"""
        try:
            # Загружаем профили клиентов
            self.clients_df = pd.read_csv('clients.csv')
            
            # Загружаем транзакции
            transaction_files = glob.glob('client_*_transactions_3m.csv')
            all_transactions = []
            
            for file in transaction_files:
                df = pd.read_csv(file)
                all_transactions.append(df)
            
            self.transactions_df = pd.concat(all_transactions, ignore_index=True)
            self.transactions_df['date'] = pd.to_datetime(self.transactions_df['date'])
            
            # Загружаем переводы
            transfer_files = glob.glob('client_*_transfers_3m.csv')
            all_transfers = []
            
            for file in transfer_files:
                df = pd.read_csv(file)
                all_transfers.append(df)
            
            self.transfers_df = pd.concat(all_transfers, ignore_index=True)
            self.transfers_df['date'] = pd.to_datetime(self.transfers_df['date'])
            
            return True
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False
    
    def analyze_client_behavior(self, client_code):
        """Анализирует поведение конкретного клиента"""
        client_info = self.clients_df[self.clients_df['client_code'] == client_code].iloc[0]
        
        # Получаем транзакции клиента
        client_transactions = self.transactions_df[
            self.transactions_df['client_code'] == client_code
        ].copy()
        
        # Получаем переводы клиента
        client_transfers = self.transfers_df[
            self.transfers_df['client_code'] == client_code
        ].copy()
        
        # Анализ трат по категориям
        category_spending = client_transactions.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Анализ переводов
        transfer_analysis = client_transfers.groupby('type')['amount'].sum()
        
        # Анализ валютных операций
        fx_operations = client_transfers[client_transfers['type'].isin(['fx_buy', 'fx_sell'])]
        fx_volume = fx_operations['amount'].sum() if not fx_operations.empty else 0
        
        # Анализ онлайн-сервисов
        online_services = client_transactions[
            client_transactions['category'].isin(['Едим дома', 'Смотрим дома', 'Играем дома'])
        ]['amount'].sum()
        
        # Анализ путешествий
        travel_spending = client_transactions[
            client_transactions['category'].isin(['Путешествия', 'Такси', 'Отели'])
        ]['amount'].sum()
        
        # Анализ ресторанов и люкса
        premium_spending = client_transactions[
            client_transactions['category'].isin(['Кафе и рестораны', 'Косметика и Парфюмерия', 'Ювелирные украшения'])
        ]['amount'].sum()
        
        return {
            'client_info': client_info.to_dict(),
            'category_spending': category_spending.to_dict(),
            'transfer_analysis': transfer_analysis.to_dict(),
            'fx_volume': fx_volume,
            'online_services': online_services,
            'travel_spending': travel_spending,
            'premium_spending': premium_spending,
            'total_spending': client_transactions['amount'].sum(),
            'avg_monthly_balance': client_info['avg_monthly_balance_KZT']
        }
    
    def calculate_product_benefits(self, client_analysis):
        """Рассчитывает выгоду от каждого продукта для клиента"""
        benefits = {}
        client_info = client_analysis['client_info']
        
        # Карта для путешествий
        travel_benefit = client_analysis['travel_spending'] * 0.04
        benefits['Карта для путешествий'] = {
            'benefit': travel_benefit,
            'reason': f"Кешбэк 4% на путешествия: {travel_benefit:,.0f} ₸"
        }
        
        # Премиальная карта
        base_benefit = client_analysis['total_spending'] * 0.02
        premium_benefit = client_analysis['premium_spending'] * 0.04
        atm_savings = 0  # Можно рассчитать на основе atm_withdrawal
        
        if client_info['avg_monthly_balance_KZT'] > 1000000:
            tier_multiplier = 1.5
        else:
            tier_multiplier = 1.0
            
        premium_total = (base_benefit + premium_benefit) * tier_multiplier
        benefits['Премиальная карта'] = {
            'benefit': premium_total,
            'reason': f"Кешбэк {2*tier_multiplier}% на все + 4% на люкс: {premium_total:,.0f} ₸"
        }
        
        # Кредитная карта
        top_categories = list(client_analysis['category_spending'].keys())[:3]
        favorite_benefit = sum([
            client_analysis['category_spending'].get(cat, 0) * 0.10 
            for cat in top_categories
        ])
        online_benefit = client_analysis['online_services'] * 0.10
        credit_total = favorite_benefit + online_benefit
        
        benefits['Кредитная карта'] = {
            'benefit': credit_total,
            'reason': f"10% в топ-категориях + онлайн: {credit_total:,.0f} ₸"
        }
        
        # Обмен валют
        fx_benefit = client_analysis['fx_volume'] * 0.005  # 0.5% экономии на спреде
        benefits['Обмен валют'] = {
            'benefit': fx_benefit,
            'reason': f"Экономия на обмене: {fx_benefit:,.0f} ₸"
        }
        
        # Депозиты (на основе остатка)
        balance = client_info['avg_monthly_balance_KZT']
        if balance > 500000:
            deposit_benefit = balance * 0.12 / 12  # 12% годовых
            benefits['Депозит сберегательный'] = {
                'benefit': deposit_benefit,
                'reason': f"Доход с депозита: {deposit_benefit:,.0f} ₸/мес"
            }
        
        # Инвестиции
        if balance > 100000:
            invest_benefit = balance * 0.08 / 12  # 8% годовых
            benefits['Инвестиции'] = {
                'benefit': invest_benefit,
                'reason': f"Потенциальный доход: {invest_benefit:,.0f} ₸/мес"
            }
        
        return benefits
    
    def generate_recommendation(self, client_code):
        """Генерирует рекомендацию для клиента"""
        analysis = self.analyze_client_behavior(client_code)
        benefits = self.calculate_product_benefits(analysis)
        
        # Выбираем продукт с максимальной выгодой
        if not benefits:
            return None
            
        best_product = max(benefits.keys(), key=lambda x: benefits[x]['benefit'])
        best_benefit = benefits[best_product]
        
        # Генерируем персонализированное сообщение
        client_name = analysis['client_info']['name']
        message = self.generate_push_message(client_name, best_product, analysis, best_benefit)
        
        return {
            'client_code': client_code,
            'product': best_product,
            'push_notification': message,
            'benefit': best_benefit['benefit'],
            'reason': best_benefit['reason']
        }
    
    def generate_push_message(self, name, product, analysis, benefit_info):
        """Генерирует персонализированное продающее пуш-уведомление"""
        
        # Получаем детальные данные для персонализации
        travel_spending = analysis.get('travel_spending', 0)
        premium_spending = analysis.get('premium_spending', 0)
        online_services = analysis.get('online_services', 0)
        fx_volume = analysis.get('fx_volume', 0)
        balance = analysis.get('avg_monthly_balance', 0)
        top_categories = list(analysis.get('category_spending', {}).keys())[:3]
        
        templates = {
            "Карта для путешествий": f"{name}, анализ ваших трат показал: вы потратили {travel_spending:,.0f} ₸ на путешествия и такси за 3 месяца! Это {travel_spending/analysis.get('total_spending', 1)*100:.1f}% от всех трат. С картой для путешествий вы вернете {benefit_info['benefit']:,.0f} ₸ кешбэка + бесплатные снятия в аэропортах. Оформите за 2 минуты!",
            
            "Премиальная карта": f"{name}, наш анализ выявил: у вас баланс {balance:,.0f} ₸ и траты на люкс {premium_spending:,.0f} ₸ за период. Вы идеальный кандидат для премиальной карты! 3% кешбэк + VIP-привилегии вернут {benefit_info['benefit']:,.0f} ₸/мес. Станьте VIP-клиентом прямо сейчас!",
            
            "Кредитная карта": f"{name}, мы проанализировали ваши траты и обнаружили топ-категории: {', '.join(top_categories)}. Вы тратите {analysis.get('total_spending', 0):,.0f} ₸ в этих категориях! Кредитная карта с 10% кешбэком в любимых категориях вернет {benefit_info['benefit']:,.0f} ₸/мес. Оформите онлайн за 3 минуты!",
            
            "Обмен валют": f"{name}, анализ ваших операций показал: вы обмениваете валюту на {fx_volume:,.0f} ₸ за 3 месяца! Наш обмен валют с лучшим курсом в городе сэкономит {benefit_info['benefit']:,.0f} ₸. Настройте авто-обмен по целевому курсу и получайте уведомления о выгодных курсах!",
            
            "Депозит сберегательный": f"{name}, у вас стабильный баланс {balance:,.0f} ₸ - идеально для депозита! Сберегательный вклад с 12% годовых принесет {benefit_info['benefit']:,.0f} ₸/мес пассивного дохода. Не теряйте деньги на инфляции - откройте вклад за 1 минуту!",
            
            "Инвестиции": f"{name}, анализ вашего профиля показал готовность к инвестициям: баланс {balance:,.0f} ₸, активность {len(analysis.get('category_spending', {}))} категорий трат. Инвестиционный счет с 8% доходностью принесет {benefit_info['benefit']:,.0f} ₸/мес. Начните с 10,000 ₸ - откройте счет за 5 минут!"
        }
        
        return templates.get(product, f"{name}, наш анализ вашего поведения показал идеальный продукт для вас. Узнайте подробнее в приложении BCC Bank!")

# Инициализируем анализаторы
analyzer = ClientAnalyzer()
ml_analyzer = MLBankAnalyzer()
ml_initialized = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/clients')
def get_clients():
    if analyzer.clients_df is None:
        if not analyzer.load_data():
            return jsonify({'error': 'Ошибка загрузки данных'}), 500
    
    clients = analyzer.clients_df.to_dict('records')
    return jsonify(clients)

@app.route('/api/analyze/<int:client_code>')
def analyze_client(client_code):
    if analyzer.clients_df is None:
        if not analyzer.load_data():
            return jsonify({'error': 'Ошибка загрузки данных'}), 500
    
    try:
        recommendation = analyzer.generate_recommendation(client_code)
        if recommendation:
            return jsonify(recommendation)
        else:
            return jsonify({'error': 'Не удалось сгенерировать рекомендацию'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_all')
def generate_all_recommendations():
    if analyzer.clients_df is None:
        if not analyzer.load_data():
            return jsonify({'error': 'Ошибка загрузки данных'}), 500
    
    recommendations = []
    for client_code in analyzer.clients_df['client_code']:
        try:
            rec = analyzer.generate_recommendation(client_code)
            if rec:
                recommendations.append(rec)
        except Exception as e:
            print(f"Ошибка для клиента {client_code}: {e}")
    
    return jsonify(recommendations)

@app.route('/api/export_csv')
def export_csv():
    if analyzer.clients_df is None:
        if not analyzer.load_data():
            return jsonify({'error': 'Ошибка загрузки данных'}), 500
    
    recommendations = []
    for client_code in analyzer.clients_df['client_code']:
        try:
            rec = analyzer.generate_recommendation(client_code)
            if rec:
                recommendations.append({
                    'client_code': rec['client_code'],
                    'product': rec['product'],
                    'push_notification': rec['push_notification']
                })
        except Exception as e:
            print(f"Ошибка для клиента {client_code}: {e}")
    
    # Создаем CSV
    df = pd.DataFrame(recommendations)
    csv_content = df.to_csv(index=False, encoding='utf-8-sig')
    
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=recommendations.csv'
    }

@app.route('/api/ml/init')
def init_ml():
    """Инициализирует ML анализатор"""
    global ml_initialized
    try:
        if not ml_analyzer.load_data():
            return jsonify({'error': 'Ошибка загрузки данных'}), 500
        
        ml_analyzer.create_advanced_features()
        ml_analyzer.perform_clustering()
        accuracy = ml_analyzer.train_product_classifier()
        
        ml_initialized = True
        
        return jsonify({
            'success': True,
            'message': 'ML анализатор инициализирован',
            'accuracy': accuracy,
            'features_count': len(ml_analyzer.features_df.columns),
            'clusters_count': len(set(ml_analyzer.clusters))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/recommendations')
def get_ml_recommendations():
    """Получает ML рекомендации"""
    global ml_initialized
    if not ml_initialized:
        return jsonify({'error': 'ML анализатор не инициализирован'}), 400
    
    try:
        recommendations = ml_analyzer.generate_ml_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/analyze/<int:client_code>')
def analyze_client_ml(client_code):
    """Анализирует клиента с помощью ML"""
    global ml_initialized
    if not ml_initialized:
        return jsonify({'error': 'ML анализатор не инициализирован'}), 400
    
    try:
        # Получаем данные клиента
        client_data = ml_analyzer.features_df[ml_analyzer.features_df['client_code'] == client_code]
        if client_data.empty:
            return jsonify({'error': 'Клиент не найден'}), 404
        
        client = client_data.iloc[0]
        
        # Предсказываем продукт
        feature_columns = [col for col in ml_analyzer.features_df.columns if col not in ['client_code', 'cluster']]
        X = client[feature_columns].fillna(0).values.reshape(1, -1)
        
        predicted_product = ml_analyzer.models['classifier'].predict(X)[0]
        prediction_proba = ml_analyzer.models['classifier'].predict_proba(X)[0]
        confidence = max(prediction_proba)
        
        # Получаем важность признаков для этого клиента
        feature_importance = ml_analyzer.feature_importance.head(10).to_dict('records')
        
        # Получаем данные о кластере
        cluster_id = client['cluster']
        cluster_insights = ml_analyzer.get_cluster_insights()
        cluster_info = next((c for c in cluster_insights if c['cluster_id'] == cluster_id), None)
        
        return jsonify({
            'client_code': client_code,
            'predicted_product': predicted_product,
            'confidence': confidence,
            'cluster': cluster_info,
            'feature_importance': feature_importance,
            'client_features': {
                'age': client['age'],
                'balance': client['balance'],
                'travel_ratio': client['travel_ratio'],
                'online_ratio': client['online_ratio'],
                'luxury_ratio': client['luxury_ratio'],
                'savings_index': client['savings_index'],
                'activity_index': client['activity_index']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/insights')
def get_ml_insights():
    """Получает инсайты по кластерам"""
    global ml_initialized
    if not ml_initialized:
        return jsonify({'error': 'ML анализатор не инициализирован'}), 400
    
    try:
        insights = ml_analyzer.get_cluster_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml/export')
def export_ml_csv():
    """Экспортирует ML рекомендации в CSV"""
    global ml_initialized
    if not ml_initialized:
        return jsonify({'error': 'ML анализатор не инициализирован'}), 400
    
    try:
        recommendations = ml_analyzer.generate_ml_recommendations()
        df = pd.DataFrame(recommendations)
        csv_content = df.to_csv(index=False, encoding='utf-8-sig')
        
        return csv_content, 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=ml_recommendations.csv'
        }
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
