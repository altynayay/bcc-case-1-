import pandas as pd

# Проверяем CSV файл
print("🔍 Проверка CSV файла с кириллицей")
print("="*50)

# Читаем CSV
df = pd.read_csv('ml_recommendations_demo.csv', encoding='utf-8-sig')

print(f"✅ Файл успешно прочитан")
print(f"📊 Размер: {df.shape[0]} строк, {df.shape[1]} колонок")
print(f"📝 Колонки: {list(df.columns)}")

print("\n🎯 Примеры данных:")
print("-" * 30)

# Показываем первые 3 строки с кириллицей
for i, row in df.head(3).iterrows():
    print(f"\n{i+1}. {row['name']} - {row['product']}")
    print(f"   Сообщение: {row['push_notification'][:100]}...")

print("\n📈 Статистика:")
print(f"   Уникальных клиентов: {df['name'].nunique()}")
print(f"   Уникальных продуктов: {df['product'].nunique()}")
print(f"   Средняя уверенность: {df['confidence'].mean():.1%}")
print(f"   Средняя выгода: {df['benefit'].mean():,.0f} ₸")

# Проверяем, есть ли проблемы с кодировкой
print("\n🔍 Проверка кодировки:")
problematic_chars = []
for col in df.columns:
    if df[col].dtype == 'object':  # текстовые колонки
        for val in df[col].dropna():
            if isinstance(val, str) and any(ord(c) > 127 for c in val):
                # Проверяем, что кириллица отображается правильно
                if '' in val or '?' in val:
                    problematic_chars.append(f"{col}: {val[:50]}...")

if problematic_chars:
    print("❌ Найдены проблемы с кодировкой:")
    for problem in problematic_chars[:5]:  # показываем первые 5
        print(f"   {problem}")
else:
    print("✅ Кодировка в порядке - кириллица отображается корректно")

print("\n💾 Файл готов для открытия в Excel!")
print("   - Используйте кодировку UTF-8")
print("   - Разделитель: запятая")
print("   - Кириллица должна отображаться корректно")
