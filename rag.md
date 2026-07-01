


Теперь у меня есть полная картина структуры репозитория `rust-unofficial/patterns` и содержание ключевых файлов. Давайте подготовим промпт для OpenCode.

---

## Промпт для OpenCode

```markdown
# Задача: RAG-агент для сравнения ответов LLM с/без контекста

## Контекст

У нас есть существующий C# проект (Program.cs), который:
- Индексирует markdown-файлы из папки `patterns` (клон https://github.com/rust-unofficial/patterns/tree/main/src)
- Создает векторное хранилище в SQLite с эмбеддингами через Ollama (nomic-embed-text)
- Поддерживает две стратегии chunking: FixedSize и Structural
- Реализует семантический поиск по чанкам

## Цель

Расширить проект до полноценного RAG-агента с двумя режимами работы:
1. **Без RAG** — прямой запрос к LLM (Claude 3.5 Sonnet)
2. **С RAG** — поиск релевантных чанков → объединение с вопросом → запрос к LLM

Сравнить качество ответов на 10 контрольных вопросах.

## Технические требования

- **C# 14 / .NET 10**
- **Anthropic C# SDK 12.24.1** (NuGet: `Anthropic.SDK`)
- **Модель**: `claude-3-5-sonnet-20240620`
- **Chunking**: использовать только `StructuralChunkingStrategy` (FixedSize оставить для сравнения индексации, но для RAG использовать Structural)
- **Эмбеддинги**: Ollama + nomic-embed-text (уже реализовано)
- **База**: SQLite (уже реализовано)

## Структура репозитория patterns

Папка `patterns/src` содержит mdbook-проект "Rust Design Patterns" со следующей структурой:

```
src/
├── intro.md
├── translations.md
├── idioms/
│   ├── index.md
│   ├── coercion-arguments.md      # Использование заимствованных типов
│   ├── concat-format.md           # Конкатенация строк через format!
│   ├── ctor.md                    # Конструкторы (new + Default)
│   ├── default.md                 # Трейт Default
│   ├── deref.md                   # Коллекции как умные указатели
│   ├── dtor-finally.md            # Финализация в деструкторах
│   ├── ffi/                       # FFI идиомы
│   ├── mem-replace.md             # mem::take/mem::replace
│   ├── on-stack-dyn-dispatch.md   # Динамическая диспатч на стеке
│   ├── option-iter.md             # Итерация по Option
│   ├── pass-var-to-closure.md     # Передача переменных в замыкания
│   ├── priv-extend.md             # Приватность для расширяемости
│   ├── rustdoc-init.md            # Инициализация для документации
│   ├── temporary-mutability.md    # Временная изменяемость
│   └── return-consumed-arg-on-error.md
├── patterns/
│   ├── index.md
│   ├── behavioural/
│   │   ├── intro.md
│   │   ├── command.md
│   │   ├── interpreter.md
│   │   ├── newtype.md             # Паттерн Newtype
│   │   ├── RAII.md                # RAII Guards
│   │   ├── strategy.md            # Паттерн Strategy
│   │   └── visitor.md             # Паттерн Visitor
│   ├── creational/
│   │   ├── intro.md
│   │   ├── builder.md             # Паттерн Builder
│   │   └── fold.md                # Паттерн Fold
│   ├── structural/
│   │   ├── intro.md
│   │   ├── compose-structs.md     # Декомпозиция структур
│   │   ├── small-crates.md        # Предпочтение маленьких крейтов
│   │   ├── trait-for-bounds.md    # Кастомные трейты для bounds
│   │   └── unsafe-mods.md         # Изоляция unsafe
│   └── ffi/
│       ├── intro.md
│       ├── export.md              # Object-Based APIs
│       └── wrappers.md            # Type Consolidation
├── anti_patterns/
│   ├── index.md
│   ├── borrow_clone.md            # Клонирование для borrow checker
│   ├── deny-warnings.md           # #[deny(warnings)]
│   └── deref.md                   # Deref полиморфизм
├── functional/
│   ├── index.md
│   ├── paradigms.md               # Парадигмы программирования
│   ├── generics-type-classes.md   # Generics как type classes
│   └── optics.md                  # Функциональная оптика
└── additional_resources/
├── index.md
└── design-principles.md
```

## Что нужно реализовать

### 1. Сервис для запросов к Anthropic Claude

```csharp
public interface ILlmService
{
    Task<string> AskAsync(string prompt, string? systemPrompt = null, CancellationToken ct = default);
}

public class AnthropicLlmService : ILlmService
{
    // Использовать Anthropic.SDK 12.24.1
    // Модель: claude-3-5-sonnet-20240620
    // Максимум токенов: 1024
    // Temperature: 0.1 (для воспроизводимости)
}
```

### 2. RAG Pipeline

```csharp
public class RagPipeline
{
    // 1. Получить эмбеддинг вопроса через OllamaEmbeddingService
    // 2. Найти top-K (K=3) релевантных чанков через SqliteVectorStore (только Structural стратегия)
    // 3. Сформировать контекст из найденных чанков:
    //    - Заголовок файла
    //    - Секция (если есть)
    //    - Содержимое чанка
    // 4. Сформировать промпт:
    //    System: "You are an expert in Rust design patterns. Answer based ONLY on the provided context."
    //    User: контекст + вопрос
    // 5. Отправить в LLM и вернуть ответ
}
```

### 3. Два режима работы

```csharp
public enum AgentMode { WithoutRag, WithRag }

public class ComparisonAgent
{
    public async Task<ComparisonResult> CompareAsync(string question, AgentMode mode);
}
```

### 4. 10 контрольных вопросов

Создать файл `test-questions.json`:

```json
[
  {
    "id": 1,
    "question": "What is the RAII guard pattern in Rust and how does it differ from traditional RAII?",
    "expected_answer": "RAII guards extend traditional RAII by using the type system to ensure access to a resource is always mediated by a guard object. The guard contains a reference to the underlying resource, and the borrow checker ensures the guard cannot outlive the resource. Classic example: MutexGuard from std library. The guard implements Deref to be used like a pointer.",
    "expected_sources": ["patterns/behavioural/RAII.md"],
    "difficulty": "medium"
  },
  {
    "id": 2,
    "question": "How does the Builder pattern help when a struct has many optional fields in Rust?",
    "expected_answer": "Builder pattern separates construction from representation. In Rust it's especially useful because Rust lacks function overloading and default parameters. The builder allows step-by-step construction, keeps client code backwards compatible when adding fields, and can be used as a template for constructing multiple objects.",
    "expected_sources": ["patterns/creational/builder.md"],
    "difficulty": "medium"
  },
  {
    "id": 3,
    "question": "What is the 'Clone to satisfy the borrow checker' anti-pattern and why is it problematic?",
    "expected_answer": "This anti-pattern occurs when developers use .clone() to resolve borrow checker errors without understanding ownership. It creates unsynchronized copies of data. While sometimes acceptable (prototypes, non-performance-critical code), it indicates a lack of understanding of Rust's ownership model. Rc and Arc are exceptions as they handle clones intelligently via reference counting.",
    "expected_sources": ["anti_patterns/borrow_clone.md"],
    "difficulty": "easy"
  },
  {
    "id": 4,
    "question": "Explain the mem::take and mem::replace idiom for changing enum variants without cloning.",
    "expected_answer": "When you have &mut MyEnum and want to change variant while keeping owned values (like String), mem::take swaps the value with its Default (empty String for String, which doesn't allocate), returning the previous owned value. mem::replace allows specifying the replacement value. This avoids the 'clone to satisfy borrow checker' anti-pattern. For Option, Option::take() is preferred.",
    "expected_sources": ["idioms/mem-replace.md", "anti_patterns/borrow_clone.md"],
    "difficulty": "hard"
  },
  {
    "id": 5,
    "question": "What is the Newtype pattern and what are its primary use cases in Rust?",
    "expected_answer": "Newtype is a tuple struct with a single field creating an opaque wrapper. Use cases: type safety (distinguishing units like Miles vs Kilometers), encapsulation (hiding implementation details), restricting functionality, making Copy types have move semantics, overriding trait implementations (e.g., custom Display for passwords). It's a zero-cost abstraction.",
    "expected_sources": ["patterns/behavioural/newtype.md"],
    "difficulty": "medium"
  },
  {
    "id": 6,
    "question": "How can you use destructors for finalization instead of finally blocks in Rust?",
    "expected_answer": "Since Rust lacks finally blocks, you can create a struct implementing Drop and instantiate it at the start of a function. The destructor runs on all exit paths: normal return, early return, ? operator, and panics. Important caveats: destructors aren't guaranteed in infinite loops or double-panics. Variable must start with _ (but not just _) to avoid unused warnings. Must not be moved or returned.",
    "expected_sources": ["idioms/dtor-finally.md", "patterns/behavioural/RAII.md"],
    "difficulty": "medium"
  },
  {
    "id": 7,
    "question": "Why should function arguments prefer borrowed types like &str over &String?",
    "expected_answer": "Using borrowed types (&str, &[T], &T) over borrowing owned types (&String, &Vec<T>, &Box<T>) increases flexibility through deref coercion. &String has two layers of indirection while &str has one. Functions taking &str accept &String, string literals, and slices. This follows the principle: prefer the borrowed type over borrowing the owned type.",
    "expected_sources": ["idioms/coercion-arguments.md"],
    "difficulty": "easy"
  },
  {
    "id": 8,
    "question": "What is the Strategy pattern and how can it be implemented without traits in Rust?",
    "expected_answer": "Strategy pattern separates algorithm skeleton from specific implementations, enabling dependency inversion. In Rust it can be implemented with traits (Formatter trait with Text/Json implementations) or with closures/closures as strategies. Rust's Option::map is an example of strategy pattern with closures. Serde is a real-world example allowing format swapping (serde_json vs serde_cbor).",
    "expected_sources": ["patterns/behavioural/strategy.md"],
    "difficulty": "medium"
  },
  {
    "id": 9,
    "question": "Explain struct decomposition for independent borrowing and when it is useful.",
    "expected_answer": "When a large struct causes borrow checker issues (whole struct used at once preventing independent field borrowing), decompose it into smaller structs and compose them back. Each smaller struct can be borrowed independently. This often reveals better design with smaller units of functionality. Caveat: can lead to verbose code and worse abstractions if overused.",
    "expected_sources": ["patterns/structural/compose-structs.md"],
    "difficulty": "hard"
  },
  {
    "id": 10,
    "question": "Why is #[deny(warnings)] considered an anti-pattern in Rust?",
    "expected_answer": "It breaks Rust's stability guarantees. New compiler versions may introduce new warnings (e.g., for deprecated APIs or upcoming breaking changes), causing builds to fail. It also prevents using tools like clippy that add new lints. Alternatives: use RUSTFLAGS='-D warnings' in CI, or explicitly list safe lints to deny (bad_style, dead_code, etc.) without denying all warnings.",
    "expected_sources": ["anti_patterns/deny-warnings.md"],
    "difficulty": "easy"
  }
]
```

### 5. Сравнение и отчет

```csharp
public class EvaluationEngine
{
    // Для каждого вопроса:
    // 1. Получить ответ без RAG
    // 2. Получить ответ с RAG
    // 3. Сохранить результаты в SQLite (таблица evaluations)
    // 4. Вывести сравнительную таблицу в консоль
    
    // Метрики для сравнения:
    // - Длина ответа
    // - Упоминание ожидаемых ключевых концепций (простая эвристика)
    // - Источники, использованные в RAG-режиме
    // - Время ответа
}
```

### 6. Схема БД (дополнить существующую)

```sql
CREATE TABLE evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    mode TEXT NOT NULL, -- 'without_rag' | 'with_rag'
    answer TEXT NOT NULL,
    sources_used TEXT, -- JSON array для RAG
    response_time_ms INTEGER,
    created_at TEXT NOT NULL
);
```

## Архитектура решения

```
┌─────────────────────────────────────────────────────────────┐
│                      ComparisonAgent                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  RagPipeline │  │ AnthropicLlm │  │ EvaluationEngine│   │
│  │  (search+ctx)│  │    Service   │  │                 │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  SqliteVectorStore (уже есть) + новая таблица        │   │
│  │  OllamaEmbeddingService (уже есть)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Требования к коду

1. **Использовать существующие классы** из Program.cs (не переписывать их)
2. **Добавить новые файлы**:
    - `AnthropicLlmService.cs`
    - `RagPipeline.cs`
    - `ComparisonAgent.cs`
    - `EvaluationEngine.cs`
    - `Models.cs` (дополнительные record-типы)
    - `test-questions.json` (embedded resource)
3. **Обновить Program.cs**:
    - Добавить выбор режима: `index` (существующий), `compare` (новый), `demo` (существующий поиск)
    - В режиме `compare`: загрузить вопросы, прогнать оба режима, вывести отчет
4. **Обработка ошибок**: graceful degradation если Anthropic API недоступен (выводить предупреждение, пропускать сравнение)
5. **Конфигурация**:
    - `ANTHROPIC_API_KEY` — env var для API ключа
    - `ANTHROPIC_MODEL` — env var для модели (default: claude-3-5-sonnet-20240620)
    - `RAG_TOP_K` — количество чанков для RAG (default: 3)

## Формат вывода отчета

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RAG vs NO-RAG COMPARISON REPORT                           ║
╠════╦═════════════════════════════════════════════════════════════════════════╣
║ Q1 ║ What is the RAII guard pattern...                                       ║
╠════╬═════════════════════════════════════════════════════════════════════════╣
║    ║ WITHOUT RAG:                                                            ║
║    ║ [ответ...]                                                              ║
║    ║ Length: 245 chars | Time: 1.2s                                          ║
╠════╬═════════════════════════════════════════════════════════════════════════╣
║    ║ WITH RAG:                                                               ║
║    ║ Sources: patterns/behavioural/RAII.md                                   ║
║    ║ [ответ...]                                                              ║
║    ║ Length: 312 chars | Time: 1.5s                                          ║
╠════╬═════════════════════════════════════════════════════════════════════════╣
║    ║ Analysis: RAG ответ точнее упоминает MutexGuard и borrow checker        ║
╚════╩═════════════════════════════════════════════════════════════════════════╝
```

## Важные замечания

- **Не удаляйте** существующую функциональность индексации и демо-поиска
- **Anthropic SDK 12.24.1** — проверьте актуальный API (вероятно, `AnthropicClient` или `AnthropicApi`)
- **Prompt engineering**: для RAG-режима явно укажите "Answer based ONLY on the provided context. If the context doesn't contain the answer, say so."
- **Для режима без RAG**: системный промпт "You are an expert in Rust design patterns." без контекста
- **Кэширование**: можно кэшировать ответы в SQLite чтобы не тратить токены при повторных запусках

## Проверка перед отправкой

- [ ] Код компилируется (C# 14)
- [ ] Все 10 вопросов обрабатываются в обоих режимах
- [ ] Отчет выводится в консоль
- [ ] Результаты сохраняются в БД
- [ ] Обработка отсутствия ANTHROPIC_API_KEY (warning + skip)
```

---

## Обоснование архитектурных решений в промпте

**Почему Anthropic SDK 12.24.1**: Это актуальная стабильная версия C# SDK для Claude. Модель `claude-3-5-sonnet-20240620` выбрана как оптимальный баланс качества/скорости/цены для технических вопросов.

**Почему 10 вопросов**: Покрывают все основные разделы книги:
- 3 вопроса по Idioms (mem-replace, dtor-finally, coercion-arguments)
- 3 вопроса по Patterns (RAII, Builder, Newtype, Strategy, Compose-structs)
- 2 вопроса по Anti-patterns (borrow_clone, deny-warnings)
- 1 вопрос по Functional (paradigms опосредованно)
- Разный уровень сложности (easy/medium/hard)

**Почему StructuralChunkingStrategy**: Для markdown-файлов с четкой структурой (заголовки, секции) структурный chunking дает лучшее качество поиска — чанки соответствуют логическим блокам, а не обрываются посреди предложения.

**Почему K=3**: Оптимальный баланс между полнотой контекста и размером промпта. Три чанка обычно покрывают 1-2 паттерна полностью.

**Почему temperature=0.1**: Для воспроизводимости сравнения. Высокая температура даст разные ответы при каждом запуске, что затруднит объективное сравнение.

**Почему кэширование**: API вызовы стоят денег и занимают время. При разработке и отладке кэш позволяет многократно перезапускать без повторных расходов.