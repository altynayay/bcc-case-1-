import pandas as pd

# Загружаем результаты
df = pd.read_csv('ml_recommendations_demo.csv')

print("🎯 ПРИМЕРЫ ПЕРСОНАЛИЗИРОВАННЫХ СООБЩЕНИЙ")
print("="*80)

for i, row in df.head(5).iterrows():
    print(f"\n{i+1}. {row['name']} - {row['product']}")
    print(f"   Уверенность: {row['confidence']:.1%}")
    print(f"   Выгода: {row['benefit']:,.0f} ₸")
    print(f"   Кластер: {row['cluster']}")
    print(f"   Сообщение: {row['push_notification']}")
    print("-" * 80)

print(f"\n📊 Всего рекомендаций: {len(df)}")
print(f"🎯 Средняя уверенность: {df['confidence'].mean():.1%}")
print(f"💰 Средняя выгода: {df['benefit'].mean():,.0f} ₸")
