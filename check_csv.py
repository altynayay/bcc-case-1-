import pandas as pd
import chardet

# Проверяем кодировку файла
with open('ml_recommendations_demo.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(f"Определенная кодировка: {result}")

# Пробуем разные кодировки
encodings = ['utf-8', 'cp1251', 'latin1', 'utf-8-sig']

for encoding in encodings:
    try:
        df = pd.read_csv('ml_recommendations_demo.csv', encoding=encoding)
        print(f"\n✅ Успешно прочитано с кодировкой: {encoding}")
        print("Первые 3 строки:")
        print(df.head(3)[['name', 'product', 'push_notification']].to_string())
        break
    except Exception as e:
        print(f"❌ Ошибка с кодировкой {encoding}: {e}")

# Проверяем содержимое файла
print("\n" + "="*50)
print("Сырое содержимое файла (первые 500 символов):")
with open('ml_recommendations_demo.csv', 'rb') as f:
    content = f.read(500)
    print(content)
