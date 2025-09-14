import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

class MLBankAnalyzer:
    def __init__(self):
        self.clients_df = None
        self.transactions_df = None
        self.transfers_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.clusters = None
        self.feature_importance = None
        
        # Продукты для рекомендаций
        self.products = {
            "Карта для путешествий": {"categories": ["Путешествия", "Такси", "Отели"], "cashback": 0.04},
            "Премиальная карта": {"categories": ["Кафе и рестораны", "Косметика и Парфюмерия", "Ювелирные украшения"], "cashback": 0.03},
            "Кредитная карта": {"categories": ["Едим дома", "Смотрим дома", "Играем дома"], "cashback": 0.10},
            "Обмен валют": {"signals": ["fx_buy", "fx_sell"], "savings": 0.005},
            "Кредит наличными": {"signals": ["loan_payment_out"], "rate": 0.15},
            "Депозит мультивалютный": {"signals": ["fx_buy", "fx_sell"], "rate": 0.08},
            "Депозит сберегательный": {"signals": ["deposit_topup_out"], "rate": 0.12},
            "Депозит накопительный": {"signals": ["deposit_topup_out"], "rate": 0.10},
            "Инвестиции": {"signals": ["invest_out", "invest_in"], "rate": 0.08},
            "Золотые слитки": {"signals": ["gold_buy_out", "gold_sell_in"], "rate": 0.05}
        }
    
    def load_data(self):
        """Загружает и подготавливает данные"""
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
            
            print(f"✅ Загружено {len(self.clients_df)} клиентов")
            print(f"✅ Загружено {len(self.transactions_df)} транзакций")
            print(f"✅ Загружено {len(self.transfers_df)} переводов")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def create_advanced_features(self):
        """Создает продвинутые признаки для ML"""
        print("🔧 Создание инженерных признаков...")
        
        features_list = []
        
        for _, client in self.clients_df.iterrows():
            client_code = client['client_code']
            
            # Получаем данные клиента
            client_transactions = self.transactions_df[
                self.transactions_df['client_code'] == client_code
            ].copy()
            client_transfers = self.transfers_df[
                self.transfers_df['client_code'] == client_code
            ].copy()
            
            # === КЛАССИЧЕСКИЕ ПРИЗНАКИ ===
            features = {
                'client_code': client_code,
                'age': client['age'],
                'balance': client['avg_monthly_balance_KZT'],
                'status_encoded': self._encode_status(client['status']),
                'city_encoded': self._encode_city(client['city'])
            }
            
            # === ВРЕМЕННЫЕ ПРИЗНАКИ ===
            if not client_transactions.empty:
                client_transactions['hour'] = client_transactions['date'].dt.hour
                client_transactions['day_of_week'] = client_transactions['date'].dt.dayofweek
                client_transactions['month'] = client_transactions['date'].dt.month
                
                # Частота транзакций по времени
                features['transactions_per_day'] = len(client_transactions) / 90
                features['avg_transaction_amount'] = client_transactions['amount'].mean()
                features['transaction_std'] = client_transactions['amount'].std()
                features['max_transaction'] = client_transactions['amount'].max()
                features['min_transaction'] = client_transactions['amount'].min()
                
                # Временные паттерны
                features['weekend_ratio'] = (client_transactions['day_of_week'] >= 5).mean()
                features['night_ratio'] = ((client_transactions['hour'] >= 22) | (client_transactions['hour'] <= 6)).mean()
                features['morning_ratio'] = ((client_transactions['hour'] >= 6) & (client_transactions['hour'] <= 12)).mean()
                
                # Сезонность
                features['june_spending'] = client_transactions[client_transactions['month'] == 6]['amount'].sum()
                features['july_spending'] = client_transactions[client_transactions['month'] == 7]['amount'].sum()
                features['august_spending'] = client_transactions[client_transactions['month'] == 8]['amount'].sum()
            
            # === ПОВЕДЕНЧЕСКИЕ ПРИЗНАКИ ===
            if not client_transactions.empty:
                # Анализ категорий
                category_spending = client_transactions.groupby('category')['amount'].sum()
                features['category_diversity'] = len(category_spending)
                features['top_category_ratio'] = category_spending.max() / category_spending.sum() if category_spending.sum() > 0 else 0
                
                # Специфичные категории
                features['travel_spending'] = category_spending.get('Путешествия', 0) + category_spending.get('Такси', 0) + category_spending.get('Отели', 0)
                features['food_spending'] = category_spending.get('Продукты питания', 0) + category_spending.get('Кафе и рестораны', 0)
                features['online_spending'] = category_spending.get('Едим дома', 0) + category_spending.get('Смотрим дома', 0) + category_spending.get('Играем дома', 0)
                features['luxury_spending'] = category_spending.get('Косметика и Парфюмерия', 0) + category_spending.get('Ювелирные украшения', 0)
                
                # Коэффициенты
                total_spending = category_spending.sum()
                features['travel_ratio'] = features['travel_spending'] / total_spending if total_spending > 0 else 0
                features['food_ratio'] = features['food_spending'] / total_spending if total_spending > 0 else 0
                features['online_ratio'] = features['online_spending'] / total_spending if total_spending > 0 else 0
                features['luxury_ratio'] = features['luxury_spending'] / total_spending if total_spending > 0 else 0
            
            # === ФИНАНСОВЫЕ ПРИЗНАКИ ===
            if not client_transfers.empty:
                transfer_analysis = client_transfers.groupby('type')['amount'].sum()
                
                # Типы операций
                features['atm_withdrawals'] = transfer_analysis.get('atm_withdrawal', 0)
                features['p2p_transfers'] = transfer_analysis.get('p2p_out', 0) + transfer_analysis.get('p2p_in', 0)
                features['fx_operations'] = transfer_analysis.get('fx_buy', 0) + transfer_analysis.get('fx_sell', 0)
                features['deposit_operations'] = transfer_analysis.get('deposit_topup_out', 0) + transfer_analysis.get('deposit_topup_in', 0)
                features['investment_operations'] = transfer_analysis.get('invest_out', 0) + transfer_analysis.get('invest_in', 0)
                features['gold_operations'] = transfer_analysis.get('gold_buy_out', 0) + transfer_analysis.get('gold_sell_in', 0)
                
                # Финансовые коэффициенты
                total_transfers = transfer_analysis.sum()
                features['atm_ratio'] = features['atm_withdrawals'] / total_transfers if total_transfers > 0 else 0
                features['fx_ratio'] = features['fx_operations'] / total_transfers if total_transfers > 0 else 0
                features['investment_ratio'] = features['investment_operations'] / total_transfers if total_transfers > 0 else 0
            
            # === СТАТИСТИЧЕСКИЕ ПРИЗНАКИ ===
            if not client_transactions.empty:
                # Квантили трат
                amounts = client_transactions['amount'].values
                features['q25'] = np.percentile(amounts, 25)
                features['q50'] = np.percentile(amounts, 50)
                features['q75'] = np.percentile(amounts, 75)
                features['q90'] = np.percentile(amounts, 90)
                
                # Асимметрия и эксцесс
                features['skewness'] = stats.skew(amounts)
                features['kurtosis'] = stats.kurtosis(amounts)
                
                # Коэффициент вариации
                features['cv'] = client_transactions['amount'].std() / client_transactions['amount'].mean() if client_transactions['amount'].mean() > 0 else 0
            
            # === ML-ПРИЗНАКИ (на основе поведения) ===
            # Индекс лояльности (частота использования)
            features['loyalty_index'] = len(client_transactions) / 90  # транзакций в день
            
            # Индекс риска (волатильность трат)
            if not client_transactions.empty:
                features['risk_index'] = client_transactions['amount'].std() / client_transactions['amount'].mean() if client_transactions['amount'].mean() > 0 else 0
            
            # Индекс сбережений (соотношение баланса и трат)
            monthly_spending = client_transactions['amount'].sum() / 3  # за 3 месяца
            features['savings_index'] = client['avg_monthly_balance_KZT'] / monthly_spending if monthly_spending > 0 else 0
            
            # Индекс активности (разнообразие операций)
            features['activity_index'] = len(client_transfers['type'].unique()) if not client_transfers.empty else 0
            
            features_list.append(features)
        
        self.features_df = pd.DataFrame(features_list)
        print(f"✅ Создано {len(self.features_df.columns)} признаков для {len(self.features_df)} клиентов")
        return self.features_df
    
    def _encode_status(self, status):
        """Кодирует статус клиента"""
        if status not in self.label_encoders:
            self.label_encoders['status'] = LabelEncoder()
            self.label_encoders['status'].fit(['Студент', 'Зарплатный клиент', 'Премиальный клиент', 'Стандартный клиент'])
        return self.label_encoders['status'].transform([status])[0]
    
    def _encode_city(self, city):
        """Кодирует город"""
        if 'city' not in self.label_encoders:
            self.label_encoders['city'] = LabelEncoder()
            self.label_encoders['city'].fit(self.clients_df['city'].unique())
        return self.label_encoders['city'].transform([city])[0]
    
    def perform_clustering(self):
        """Выполняет кластеризацию клиентов"""
        print("🎯 Выполнение кластеризации клиентов...")
        
        # Подготавливаем данные для кластеризации
        feature_columns = [col for col in self.features_df.columns if col != 'client_code']
        X = self.features_df[feature_columns].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN кластеризация
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Иерархическая кластеризация
        linkage_matrix = linkage(X_scaled, method='ward')
        hierarchical_labels = fcluster(linkage_matrix, t=5, criterion='maxclust')
        
        # Выбираем лучшую кластеризацию по silhouette score
        kmeans_score = silhouette_score(X_scaled, kmeans_labels)
        dbscan_score = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
        hierarchical_score = silhouette_score(X_scaled, hierarchical_labels)
        
        scores = {
            'K-means': kmeans_score,
            'DBSCAN': dbscan_score,
            'Hierarchical': hierarchical_score
        }
        
        best_method = max(scores, key=scores.get)
        print(f"🏆 Лучший метод кластеризации: {best_method} (score: {scores[best_method]:.3f})")
        
        # Сохраняем результаты
        if best_method == 'K-means':
            self.clusters = kmeans_labels
        elif best_method == 'DBSCAN':
            self.clusters = dbscan_labels
        else:
            self.clusters = hierarchical_labels
        
        self.features_df['cluster'] = self.clusters
        
        # Анализируем кластеры
        self._analyze_clusters()
        
        return self.clusters
    
    def _analyze_clusters(self):
        """Анализирует характеристики кластеров"""
        print("\n📊 Анализ кластеров:")
        cluster_stats = self.features_df.groupby('cluster').agg({
            'age': 'mean',
            'balance': 'mean',
            'travel_ratio': 'mean',
            'online_ratio': 'mean',
            'luxury_ratio': 'mean',
            'savings_index': 'mean',
            'activity_index': 'mean'
        }).round(2)
        
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            if cluster_id == -1:  # DBSCAN outliers
                print(f"🔴 Кластер {cluster_id} (Аномалии): {len(self.features_df[self.features_df['cluster'] == cluster_id])} клиентов")
            else:
                stats = cluster_stats.loc[cluster_id]
                print(f"🟢 Кластер {cluster_id}: {len(self.features_df[self.features_df['cluster'] == cluster_id])} клиентов")
                print(f"   Возраст: {stats['age']:.1f}, Баланс: {stats['balance']:,.0f} ₸")
                print(f"   Путешествия: {stats['travel_ratio']:.1%}, Онлайн: {stats['online_ratio']:.1%}, Люкс: {stats['luxury_ratio']:.1%}")
    
    def train_product_classifier(self):
        """Обучает классификатор для предсказания лучшего продукта"""
        print("🤖 Обучение классификатора продуктов...")
        
        # Создаем целевую переменную на основе текущих правил
        y = []
        for _, client in self.features_df.iterrows():
            client_code = client['client_code']
            
            # Получаем данные клиента
            client_transactions = self.transactions_df[
                self.transactions_df['client_code'] == client_code
            ]
            client_transfers = self.transfers_df[
                self.transfers_df['client_code'] == client_code
            ]
            
            # Рассчитываем выгоду для каждого продукта
            benefits = self._calculate_benefits_ml(client, client_transactions, client_transfers)
            
            if benefits:
                best_product = max(benefits.keys(), key=lambda x: benefits[x])
                y.append(best_product)
            else:
                y.append('Премиальная карта')  # по умолчанию
        
        # Подготавливаем признаки
        feature_columns = [col for col in self.features_df.columns if col not in ['client_code', 'cluster']]
        X = self.features_df[feature_columns].fillna(0)
        
        # Обучаем модели
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        gb_score = gb.score(X_test, y_test)
        
        # Выбираем лучшую модель
        if rf_score >= gb_score:
            self.models['classifier'] = rf
            best_score = rf_score
            print(f"🏆 Лучшая модель: Random Forest (точность: {best_score:.3f})")
        else:
            self.models['classifier'] = gb
            best_score = gb_score
            print(f"🏆 Лучшая модель: Gradient Boosting (точность: {best_score:.3f})")
        
        # Сохраняем важность признаков
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.models['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Топ-10 важных признаков:")
        for i, (_, row) in enumerate(self.feature_importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']}: {row['importance']:.3f}")
        
        return best_score
    
    def _calculate_benefits_ml(self, client, transactions, transfers):
        """Рассчитывает выгоду от продуктов для ML"""
        benefits = {}
        
        if not transactions.empty:
            category_spending = transactions.groupby('category')['amount'].sum()
            total_spending = category_spending.sum()
        else:
            category_spending = pd.Series()
            total_spending = 0
        
        if not transfers.empty:
            transfer_analysis = transfers.groupby('type')['amount'].sum()
        else:
            transfer_analysis = pd.Series()
        
        # Карта для путешествий
        travel_spending = category_spending.get('Путешествия', 0) + category_spending.get('Такси', 0) + category_spending.get('Отели', 0)
        benefits['Карта для путешествий'] = travel_spending * 0.04
        
        # Премиальная карта
        premium_spending = category_spending.get('Кафе и рестораны', 0) + category_spending.get('Косметика и Парфюмерия', 0) + category_spending.get('Ювелирные украшения', 0)
        base_benefit = total_spending * 0.02
        premium_benefit = premium_spending * 0.04
        tier_multiplier = 1.5 if client['balance'] > 1000000 else 1.0
        benefits['Премиальная карта'] = (base_benefit + premium_benefit) * tier_multiplier
        
        # Кредитная карта
        top_categories = list(category_spending.keys())[:3]
        favorite_benefit = sum([category_spending.get(cat, 0) * 0.10 for cat in top_categories])
        online_benefit = (category_spending.get('Едим дома', 0) + category_spending.get('Смотрим дома', 0) + category_spending.get('Играем дома', 0)) * 0.10
        benefits['Кредитная карта'] = favorite_benefit + online_benefit
        
        # Обмен валют
        fx_volume = transfer_analysis.get('fx_buy', 0) + transfer_analysis.get('fx_sell', 0)
        benefits['Обмен валют'] = fx_volume * 0.005
        
        # Депозиты
        balance = client['balance']
        if balance > 500000:
            benefits['Депозит сберегательный'] = balance * 0.12 / 12
        if balance > 100000:
            benefits['Инвестиции'] = balance * 0.08 / 12
        
        return benefits
    
    def generate_ml_recommendations(self):
        """Генерирует рекомендации с использованием ML"""
        print("🎯 Генерация ML-рекомендаций...")
        
        recommendations = []
        
        for _, client in self.features_df.iterrows():
            client_code = client['client_code']
            
            # Получаем данные клиента
            client_transactions = self.transactions_df[
                self.transactions_df['client_code'] == client_code
            ]
            client_transfers = self.transfers_df[
                self.transfers_df['client_code'] == client_code
            ]
            
            # Предсказываем продукт с помощью ML
            feature_columns = [col for col in self.features_df.columns if col not in ['client_code', 'cluster']]
            X = client[feature_columns].fillna(0).values.reshape(1, -1)
            
            predicted_product = self.models['classifier'].predict(X)[0]
            prediction_proba = self.models['classifier'].predict_proba(X)[0]
            confidence = max(prediction_proba)
            
            # Рассчитываем выгоду
            benefits = self._calculate_benefits_ml(client, client_transactions, client_transfers)
            benefit = benefits.get(predicted_product, 0)
            
            # Генерируем персонализированное сообщение
            client_name = self.clients_df[self.clients_df['client_code'] == client_code]['name'].iloc[0]
            message = self._generate_ml_message(client_name, predicted_product, client, benefit, confidence)
            
            recommendations.append({
                'client_code': client_code,
                'name': client_name,
                'product': predicted_product,
                'benefit': benefit,
                'confidence': confidence,
                'cluster': client['cluster'],
                'push_notification': message
            })
        
        return recommendations
    
    def _generate_ml_message(self, name, product, client, benefit, confidence):
        """Генерирует персонализированное продающее сообщение на основе ML анализа"""
        
        # Получаем детальные инсайты клиента
        insights = self._get_client_insights(client)
        
        # Базовые шаблоны с конкретными инсайтами ИИ
        templates = {
            "Карта для путешествий": [
                f"{name}, наш ИИ проанализировал ваши траты и обнаружил: вы тратите {insights['travel_spending']:,.0f} ₸ на путешествия и такси! Это {insights['travel_ratio']:.1%} от всех ваших трат. С картой для путешествий вы вернете {benefit:,.0f} ₸ кешбэка ежемесячно. Оформите карту за 2 минуты в приложении!",
                f"{name}, ИИ выявил ваш паттерн: {insights['travel_frequency']} поездок в месяц на такси и отели. Вы в топ-{insights['travel_percentile']}% путешественников! Специальная карта для путешествий с 4% кешбэком сэкономит вам {benefit:,.0f} ₸/мес. Начните экономить уже сегодня!",
                f"{name}, анализ ваших транзакций показал: вы активный путешественник с тратами {insights['travel_spending']:,.0f} ₸/мес. Наша карта для путешествий даст вам {benefit:,.0f} ₸ кешбэка + бесплатные снятия в аэропортах. Оформите сейчас!"
            ],
            "Премиальная карта": [
                f"{name}, наш ИИ определил вас как премиального клиента! Анализ показал: баланс {client['balance']:,.0f} ₸, траты на люкс {insights['luxury_spending']:,.0f} ₸/мес ({insights['luxury_ratio']:.1%} от всех трат). Премиальная карта с 3% кешбэком вернет вам {benefit:,.0f} ₸/мес + бесплатные снятия. Станьте VIP-клиентом!",
                f"{name}, ИИ выявил ваш статус: вы тратите {insights['luxury_spending']:,.0f} ₸ на рестораны и ювелирные изделия - это {insights['luxury_ratio']:.1%} от всех трат! Премиальная карта с повышенным кешбэком даст вам {benefit:,.0f} ₸/мес. Оформите за 1 минуту!",
                f"{name}, анализ вашего поведения показал: вы цените качество и тратите {insights['luxury_spending']:,.0f} ₸ на премиум-категории. Премиальная карта с 3% кешбэком + привилегии вернет {benefit:,.0f} ₸/мес. Не упустите выгоду!"
            ],
            "Кредитная карта": [
                f"{name}, ИИ проанализировал ваши траты и обнаружил топ-категории: {insights['top_categories']}. Вы тратите {insights['favorite_spending']:,.0f} ₸ в этих категориях! Кредитная карта с 10% кешбэком в любимых категориях вернет вам {benefit:,.0f} ₸/мес. Оформите карту онлайн за 3 минуты!",
                f"{name}, наш алгоритм выявил ваш паттерн: {insights['online_ratio']:.1%} трат онлайн ({insights['online_spending']:,.0f} ₸/мес). Кредитная карта с 10% кешбэком на онлайн-покупки сэкономит вам {benefit:,.0f} ₸/мес. Начните экономить прямо сейчас!",
                f"{name}, анализ показал: ваши любимые категории - {insights['top_categories']} с тратами {insights['favorite_spending']:,.0f} ₸/мес. Кредитная карта с максимальным кешбэком 10% вернет {benefit:,.0f} ₸/мес. Оформите за 2 клика в приложении!"
            ],
            "Обмен валют": [
                f"{name}, ИИ обнаружил вашу активность в валютных операциях: {insights['fx_volume']:,.0f} ₸ за 3 месяца! Это {insights['fx_ratio']:.1%} от всех операций. Наш обмен валют с выгодным курсом сэкономит вам {benefit:,.0f} ₸. Настройте авто-обмен по целевому курсу!",
                f"{name}, анализ ваших переводов показал: вы часто обмениваете валюту ({insights['fx_operations']} операций за 3 месяца). Специальный курс обмена сэкономит {benefit:,.0f} ₸. Получите уведомления о выгодном курсе!",
                f"{name}, ИИ выявил ваш интерес к валютным операциям: {insights['fx_volume']:,.0f} ₸ за период. Наш обмен валют с лучшим курсом в городе сэкономит {benefit:,.0f} ₸. Настройте обмен прямо сейчас!"
            ],
            "Депозит сберегательный": [
                f"{name}, наш ИИ проанализировал ваш баланс {client['balance']:,.0f} ₸ и индекс сбережений {insights['savings_index']:.1f}. Вы идеальный кандидат для депозита! Сберегательный вклад с 12% годовых принесет {benefit:,.0f} ₸/мес пассивного дохода. Откройте вклад за 1 минуту!",
                f"{name}, ИИ выявил ваш потенциал: стабильный баланс {client['balance']:,.0f} ₸ и низкие траты на развлечения. Депозит с максимальной ставкой 12% даст вам {benefit:,.0f} ₸/мес. Увеличьте доходность ваших сбережений!",
                f"{name}, анализ показал: у вас есть свободные средства {client['balance']:,.0f} ₸. Сберегательный депозит с 12% годовых принесет {benefit:,.0f} ₸/мес. Не теряйте деньги на инфляции - инвестируйте в депозит!"
            ],
            "Инвестиции": [
                f"{name}, ИИ проанализировал ваш профиль и обнаружил: баланс {client['balance']:,.0f} ₸, активность {insights['activity_index']} операций, готовность к риску. Инвестиционный счет с 8% доходностью принесет {benefit:,.0f} ₸/мес. Начните инвестировать с 10,000 ₸!",
                f"{name}, наш алгоритм выявил ваш интерес к финансовым операциям: {insights['investment_operations']} инвестиционных операций за период. Инвестиционный счет с нулевыми комиссиями даст {benefit:,.0f} ₸/мес. Откройте счет за 5 минут!",
                f"{name}, анализ показал: вы готовы к инвестициям с балансом {client['balance']:,.0f} ₸. Инвестиционный портфель с 8% доходностью принесет {benefit:,.0f} ₸/мес. Начните с малого - инвестируйте 50,000 ₸!"
            ]
        }
        
        # Выбираем шаблон на основе уверенности
        if confidence > 0.8:
            template_idx = 0  # Высокая уверенность - детальные инсайты
        elif confidence > 0.6:
            template_idx = 1  # Средняя уверенность - основные инсайты
        else:
            template_idx = 2  # Низкая уверенность - базовые инсайты
        
        if product in templates:
            return templates[product][template_idx]
        else:
            return f"{name}, наш ИИ подобрал для вас идеальный продукт на основе анализа вашего поведения. Узнайте подробнее в приложении!"
    
    def _get_client_insights(self, client):
        """Получает детальные инсайты о клиенте для персонализации"""
        insights = {}
        
        # Базовые метрики с обработкой NaN
        insights['travel_spending'] = client.get('travel_spending', 0) or 0
        insights['travel_ratio'] = client.get('travel_ratio', 0) or 0
        insights['luxury_spending'] = client.get('luxury_spending', 0) or 0
        insights['luxury_ratio'] = client.get('luxury_ratio', 0) or 0
        insights['online_spending'] = client.get('online_spending', 0) or 0
        insights['online_ratio'] = client.get('online_ratio', 0) or 0
        insights['fx_volume'] = client.get('fx_operations', 0) or 0
        insights['fx_ratio'] = client.get('fx_ratio', 0) or 0
        insights['savings_index'] = client.get('savings_index', 0) or 0
        insights['activity_index'] = client.get('activity_index', 0) or 0
        
        # Дополнительные расчеты с безопасной обработкой NaN
        transactions_per_day = client.get('transactions_per_day', 0) or 0
        travel_ratio = client.get('travel_ratio', 0) or 0
        fx_ratio = client.get('fx_ratio', 0) or 0
        investment_ratio = client.get('investment_ratio', 0) or 0
        avg_transaction_amount = client.get('avg_transaction_amount', 0) or 0
        
        # Безопасное преобразование с проверкой на NaN
        try:
            insights['travel_frequency'] = max(1, int(transactions_per_day * 30 * travel_ratio))
        except (ValueError, TypeError):
            insights['travel_frequency'] = 1
            
        try:
            insights['travel_percentile'] = min(95, int(travel_ratio * 100))
        except (ValueError, TypeError):
            insights['travel_percentile'] = 5
            
        try:
            insights['fx_operations'] = max(1, int(fx_ratio * 10))
        except (ValueError, TypeError):
            insights['fx_operations'] = 1
            
        try:
            insights['investment_operations'] = max(0, int(investment_ratio * 5))
        except (ValueError, TypeError):
            insights['investment_operations'] = 0
        
        # Топ категории (симуляция на основе данных)
        if travel_ratio > 0.1:
            insights['top_categories'] = "Такси, Отели, Путешествия"
        elif insights['online_ratio'] > 0.3:
            insights['top_categories'] = "Едим дома, Смотрим дома, Играем дома"
        elif insights['luxury_ratio'] > 0.1:
            insights['top_categories'] = "Кафе и рестораны, Косметика, Ювелирные изделия"
        else:
            insights['top_categories'] = "Продукты питания, Транспорт, Услуги"
        
        try:
            insights['favorite_spending'] = max(10000, int(avg_transaction_amount * 20))
        except (ValueError, TypeError):
            insights['favorite_spending'] = 10000
        
        return insights
    
    def get_cluster_insights(self):
        """Возвращает инсайты по кластерам"""
        insights = []
        
        for cluster_id in sorted(self.features_df['cluster'].unique()):
            if cluster_id == -1:  # Пропускаем аномалии
                continue
                
            cluster_data = self.features_df[self.features_df['cluster'] == cluster_id]
            
            # Анализируем характеристики кластера
            avg_age = cluster_data['age'].mean()
            avg_balance = cluster_data['balance'].mean()
            travel_ratio = cluster_data['travel_ratio'].mean()
            online_ratio = cluster_data['online_ratio'].mean()
            luxury_ratio = cluster_data['luxury_ratio'].mean()
            
            # Определяем тип кластера
            if travel_ratio > 0.3:
                cluster_type = "Путешественники"
                recommended_products = ["Карта для путешествий", "Обмен валют"]
            elif luxury_ratio > 0.2:
                cluster_type = "Премиум клиенты"
                recommended_products = ["Премиальная карта", "Золотые слитки"]
            elif online_ratio > 0.4:
                cluster_type = "Цифровые клиенты"
                recommended_products = ["Кредитная карта", "Инвестиции"]
            elif avg_balance > 2000000:
                cluster_type = "Инвесторы"
                recommended_products = ["Депозит сберегательный", "Инвестиции", "Золотые слитки"]
            else:
                cluster_type = "Стандартные клиенты"
                recommended_products = ["Кредитная карта", "Депозит накопительный"]
            
            insights.append({
                'cluster_id': cluster_id,
                'cluster_type': cluster_type,
                'size': len(cluster_data),
                'avg_age': avg_age,
                'avg_balance': avg_balance,
                'characteristics': {
                    'travel_ratio': travel_ratio,
                    'online_ratio': online_ratio,
                    'luxury_ratio': luxury_ratio
                },
                'recommended_products': recommended_products
            })
        
        return insights

# Функция для быстрого тестирования
def quick_ml_analysis():
    """Быстрый анализ с ML для демонстрации"""
    analyzer = MLBankAnalyzer()
    
    if not analyzer.load_data():
        return None
    
    # Создаем признаки
    analyzer.create_advanced_features()
    
    # Кластеризация
    analyzer.perform_clustering()
    
    # Обучение классификатора
    accuracy = analyzer.train_product_classifier()
    
    # Генерируем рекомендации
    recommendations = analyzer.generate_ml_recommendations()
    
    # Получаем инсайты
    insights = analyzer.get_cluster_insights()
    
    return {
        'recommendations': recommendations,
        'insights': insights,
        'accuracy': accuracy,
        'feature_importance': analyzer.feature_importance.head(10).to_dict('records')
    }

if __name__ == "__main__":
    print("🚀 Запуск ML анализа BCC Bank...")
    results = quick_ml_analysis()
    
    if results:
        print(f"\n✅ Анализ завершен!")
        print(f"📊 Точность модели: {results['accuracy']:.3f}")
        print(f"🎯 Создано {len(results['recommendations'])} рекомендаций")
        print(f"🔍 Обнаружено {len(results['insights'])} кластеров")
        
        # Сохраняем результаты
        pd.DataFrame(results['recommendations']).to_csv('ml_recommendations.csv', index=False, encoding='utf-8-sig')
        print("💾 Результаты сохранены в ml_recommendations.csv")
    else:
        print("❌ Ошибка анализа")
