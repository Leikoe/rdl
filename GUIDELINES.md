# Dev Guidelines

## Error handling
- All fallible public API returns `Result<T, LayoutError>`. No `unwrap` or `expect` outside `#[cfg(test)]`.
- Validate all invariants in constructors (radix ≥ 2, permutation validity, cardinality match, etc.).

## Type design
- `Transform` is a **closed enum** — no trait objects, no `Box<dyn>`.
- All public types derive `Debug + Clone + PartialEq + Eq + Hash`.
- Constructor functions (snake_case, return `Result`) are the only way to build validated values.

## Dependencies
- Only `thiserror`. No additional crates unless approved.

## Naming
- Mirror Python module names: `math_utils`, `spaces`, `transforms`, `layout`, `builders`, `analysis`.
- Mirror Python method/function names exactly.

## Testing
- Every public function has at least one `#[test]` in a `#[cfg(test)]` block in the same file.
- Tests use small, hand-verifiable concrete examples.
