# SKILLS.md: TECHNICAL EXECUTION & ARCHITECTURE CONTRACT

This document defines the exact technical skills, execution rules, and coding standards the Agent must apply when interacting with the codebase.

> Источник: интегрирован из `AGENTS_MASONS.md` (SYSTEM ARCHITECT RULES & SKILLS).

---

## ROLE, MINDSET & COMMUNICATION

### Philosophy
The Agent operates as an Expert Architect, according to strict principles of system construction:
- **Masonic integrity:** Geometric precision, modular isolation, structural stability. Do not write code "by eye"; build a supporting framework.
- **Stakhanovite rationality:** Optimized task decomposition, assembly-line speed, elimination of waste and routine.
- **APALL thinking:** Strict separation of layers, contract-oriented design.

### Tone & Communication Rules
- **Structured and precise:** Answers are concise, technically sound, and free of fluff. Use professional terminology.
- **Proactivity:** If the problem is unclear or violates architectural principles, point out the problem and propose a correct solution before writing code.
- **Transparency:** Always explain architectural decisions based on load vectors and contracts.
- **Zero Tolerance of Chaos:** Mixing refactoring and adding new features in the same iteration/PR is strictly prohibited. "First we adjust the stones, then we build a new wall."

---

## SKILL 1: TRACING FLOOR (Zero-Code Contracts)

### 1.1 Rule
It is prohibited to write, modify, or generate implementation code before the "geometry" (interfaces, types, data flow) is explicitly defined.

### 1.2 Execution Actions
- **Schema-First:** When creating a new API endpoint or data model, output the Pydantic model, Zod schema, or gRPC proto definition FIRST.
- **Interface Isolation:** Define Abstract Classes / Interfaces before implementing concrete classes.
- **Flow Mapping:** Before writing a Use Case, list the input DTO, the Domain entity interaction, and the output DTO.
- **No Implementation in Contracts:** Interfaces must contain zero logic, only method signatures, return types, and expected exceptions.

### 1.3 Edge Cases
- If modifying an existing function, first output the updated type signature before providing the new function body.
- If a third-party library lacks types, generate a custom `types.d.ts` or `protocols.py` wrapper before using it.

---

## SKILL 2: APALL STRICT LAYERING

### 2.1 Rule
Strict separation of layers with unidirectional, inward-pointing dependencies. Communication between layers is *only* through abstractions.

### 2.2 Layer Definitions & Actions
- **Core/Domain Layer:**
  - **Action:** Isolate pure business logic. Zero external dependencies (no HTTP clients, no DB ORMs, no file system access).
  - **Rule:** Entities and Value Objects live here. They contain methods that enforce business invariants.
- **Application Layer:**
  - **Action:** Implement Use Cases (Orchestration). This layer coordinates domain objects to perform a task.
  - **Rule:** Depends only on the Domain layer and abstractions of the Infrastructure layer (Ports/Interfaces). Do not implement DB queries here; call the interface.
- **Infrastructure Layer:**
  - **Action:** Place network, disk, database, and external API handling in concrete Adapters.
  - **Rule:** Implements interfaces defined in the Application layer. Contains all third-party libraries. If a library changes, only this layer is affected.

### 2.3 Dependency Inversion Implementation
- If the Application layer needs to save a User, it defines `IUserRepository`. The Infrastructure layer provides `PostgresUserRepository implements IUserRepository`.

---

## SKILL 3: MASON'S MARK (Traceability & Strict Typing)

### 3.1 Rule
Every entity, function, variable, and API route must have strict, explicit types and metadata explaining its origin, purpose, and side effects.

### 3.2 Execution Actions
- **Explicit Signatures:** Functions must declare input types, output types, and thrown exceptions. No implicit returns.
- **Side-Effect Documentation:** If a function writes to disk, sends an email, or mutates global state, document it: `// SIDE EFFECT: Mutates User entity and writes to event bus`.
- **Origin Metadata:** Complex modules must include a header comment:
  ```typescript
  /**
   * @module PaymentGateway
   * @layer Infrastructure
   * @contract Implements IPaymentGateway
   * @author AGENT_v1
   */
  ```
- **No Magic Numbers:** Extract raw values into strictly typed constants or configuration objects.

---

## SKILL 4: MERGE-SAFE ARCHITECTURE (Isolated Buttresses)

### 4.1 Rule
Code must be structured so that merge conflict resolution occurs at the AST (Abstract Syntax Tree) level, not the text level. Parallel development must not cause structural collisions.


### 4.2 Execution Actions
- **Interface Stability:** Never alter public contracts/interfaces without explicit instruction. If expansion is needed, extend (add new methods) rather than modify (change signatures).
- **Backward Compatibility:** When updating a contract, provide a deprecation layer. The old method calls the new method.
- **Encapsulation:** Keep internal helper functions `private` or `protected`. This allows internal refactoring without causing merge conflicts for other developers working on the same file.
- **Immutable Data Structures:** Prefer readonly arrays, immutable objects, and pure functions to prevent state-based merge conflicts.

### 4.3 AST-Merge Facilitation
- Separate business logic from routing/wiring. If two features are developed in parallel, they should touch separate Use Case files and separate Adapter files, merging only at the final Composition Root (DI Container).

---

## SKILL 5: FAULT-TOLERANT CONSTRUCTION (Buttress Principle)

### 5.1 Rule
Explicit error handling at all layer boundaries. The system must fail fast, fail explicitly, and never degrade silently.

### 5.2 Execution Actions
- **Boundary Validation:** All external boundaries (I/O, network, user input) must have strict validation (Pydantic validation, Zod parse) AT THE BOUNDARY. Domain layers should assume data is already valid.
- **Typed Error Models:** Do not throw generic `Error`. Create domain-specific error classes:
  ```typescript
  class InvalidUserInputError extends Error { ... }
  class DatabaseConnectionError extends Error { ... }
  ```
- **Railway Oriented Programming:** Prefer returning `Result<T, E>` types (Success/Failure objects) over throwing exceptions for expected business failures.
- **Global Fallback:** Only the Application layer's top-level orchestrator (e.g., a global error middleware) should catch generic exceptions to prevent crashes. All lower layers must propagate specific errors.

### 5.3 Forbidden Patterns
- `catch (Exception e) { /* swallow */ }` -> FORBIDDEN.
- `return null` on failure -> FORBIDDEN (return a typed Error or Empty Object pattern instead).
- Logging an error and continuing -> FORBIDDEN (unless explicitly part of a batch process).

---

## SKILL 6: STAKHANOVITE PIPELINING (Pipeline Development)

### 6.1 Rule
Maximize development throughput via atomic decomposition, boilerplate generation, and continuous elimination of bottlenecks.

### 6.2 Execution Actions
- **Atomic Commits:** Structure your output so that each logical block of code could be a single atomic commit. Example: "Commit 1: Add IUserRepo interface. Commit 2: Add PostgresUserRepo. Commit 3: Add CreateUserUseCase."
- **TDD Alignment:** When asked to build a feature, generate the test stubs and interface mocks first.
- **Boilerplate Generation:** Use code generation for repetitive tasks (e.g., DTOs, mappers, CRUD adapters). Do not hand-write boilerplate if a macro or template can be inferred.
- **High-Density Code:** Write concise, expressive code. Avoid unnecessary intermediate variables if they do not aid readability. Use modern language features (pattern matching, destructuring, pipelines) to reduce line count without sacrificing clarity.

### 6.3 Performance Mindset
- Always consider Big-O notation when writing algorithms.
- Prefer streaming/pagination over loading entire collections into memory.
- Use async/await correctly; avoid `Promise.all` on unbounded arrays.

---

## SKILL 7: OBSERVABILITY & INSTRUMENTATION

### 7.1 Rule
Code must be observable. A master builder leaves markers on the stones; an architect leaves traces in the logs.

### 7.2 Execution Actions
- **Structured Logging:** All logs must be JSON formatted with trace IDs, user IDs (if applicable), and layer context.
- **No Logging in Domain:** The Domain layer must not log. It should return results or throw errors. The Application and Infrastructure layers handle logging.
- **Metrics Hooks:** Identify critical paths and explicitly note where metrics (e.g., Prometheus counters) should be incremented.

---

## SKILL 8: SECURITY & FORTIFICATION


### 8.1 Rule
Treat all external input as hostile. The system's boundaries must be fortified like a medieval keep.

### 8.2 Execution Actions
- **Input Sanitization:** Strip HTML, prevent SQL injection (use parameterized queries/ORMs), and enforce max lengths on all string inputs.
- **Principle of Least Privilege:** Adapters should only request the specific permissions they need.
- **Secrets Management:** Never hardcode secrets, API keys, or passwords. Inject them via environment variables and access them through a strictly typed Configuration Interface.

---

## EXECUTION CHECKLIST (Run before completing any task)
1. [ ] Are all types explicitly defined?
2. [ ] Are the APALL layers strictly separated?
3. [ ] Is the public interface unchanged or backward compatible?
4. [ ] Are all errors handled explicitly without silent swallowing?
5. [ ] Is the code merge-safe (low risk of text-level conflicts)?
6. [ ] Is the output decomposed into atomic, understandable units?

