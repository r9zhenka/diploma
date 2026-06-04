# PEFT for Text Generation Stylization

Исследование методов Parameter-Efficient Fine-Tuning для стилизации генерации текста на русскоязычных откликах в телеграм.

## Задача

Современные языковые модели генерируют текст в нейтральном, усреднённом стиле, далёком от живого общения в соцсетях. Цель — адаптировать LLM к разговорному стилю Telegram-комментариев, сравнив полное дообучение и PEFT-подходы.

## Данные

- Собственный датасет из Telegram: посты + комментарии к ним
- HF: https://huggingface.co/datasets/r9zhenka/telegram-post-comments_data
- Препроцессинг: очистка от ссылок, замена эмодзи на текст, фильтрация по длине
- Разбиение: train / val / test

## Эксперименты

| Подход | Модель | Статус |
|--------|--------|--------|
| Full fine-tuning | ruGPT-3 Small (sberbank-ai) | Обучена, 3 эпохи |
| Full fine-tuning | ruGPT-3 Medium (sberbank-ai) | Обучена, 3 эпохи |
| Full fine-tuning | Qwen2.5-0.5B | Обучена, 3 эпохи |
| LoRA | Qwen2.5-0.5B (q_proj/v_proj, r=8/16/32; +FFN/fan-вариации) | Обучена, несколько конфигураций, 8 эпох |
| (IA)^3 | Qwen2.5-0.5B (масштабирующие векторы l_k, l_v, l_f) | Обучена, 8 эпох |


## Evaluation

- BERTScore, distinct-n, perplexity
- LLM-as-a-judge для оценки стилистической адаптации

## Структура

```
├── llm_default.py              # LLM-as-a-Judge: запуск и агрегация
├── strip_outputs.py            # Чистка outputs из .ipynb перед коммитом
├── requirements.txt
├── metrics_table.csv / .md     # Итоговая сравнительная таблица
│
├── helpers_eval/               # Вспомогательные скрипты пост-обработки
│   ├── build_metrics_table.py  # Сборка большой таблицы метрик
│   ├── compute_length_stats.py # Средняя длина и combined-distinct
│   ├── count_params.py         # Аналитический подсчёт обучаемых параметров
│   └── analyse_logs.ipynb      # Разбор training-логов
│
├── evaluating/                 # Входные данные и результаты эксперимента
│   ├── eval_posts.json
│   ├── id_comments.json
│   ├── evaluation_results.json
│   ├── qwen_*.json             # Генерации по эпохам (6 PEFT-конфигов)
│   ├── qwen_*_metrics.json     # Финальные метрики обучения
│   ├── qwen_*_logs.json        # Training-логи
│   ├── judge_results*.json     # Сырые ответы LLM-судьи
│   ├── judge_stats*.json       # Агрегированная accuracy по конфигам
│   └── len_stats.json
│
├── notebooks/
│   ├── early_steps/            # Ранние наработки пайплайн
│   │   ├── eda_and_preprocess.ipynb
│   │   ├── train_val_test_split.ipynb
│   │   ├── fine_tuned_model.ipynb
│   │   ├── lora.ipynb
│   │   └── test_bert_score.ipynb
│   └── colab_books/            # Примеры ноутбуков с обучением в Colab
│       ├── finetuned_rugpt3_med_large.ipynb
│       ├── lora_v4.ipynb
│       └── lora_ffnn_v1.ipynb
│
└── README.md
```

## Основной стек

Python, PyTorch, HuggingFace Transformers, PEFT (LoRA, (IA)^3), Datasets, BERTScore, Sentence-Transformers

