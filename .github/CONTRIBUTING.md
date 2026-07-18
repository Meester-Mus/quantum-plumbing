# Contributing to Quantum Plumbing

We welcome contributions!

## How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/my-feature`)
3. **Make your changes**
4. **Write tests** for new functionality
5. **Ensure tests pass** (`pytest tests/`)
6. **Commit with clear message**
7. **Push to fork**
8. **Open pull request**

## Areas for Contribution

### High Priority
- [ ] Potential BatchNorm implementation
- [ ] Potential Dropout implementation  
- [ ] Potential Loss functions
- [ ] Network assembly (stacking layers)
- [ ] Training loops
- [ ] MNIST benchmark

### Medium Priority
- [ ] Quantum interface design
- [ ] Quantum circuit implementation
- [ ] Visualization tools
- [ ] Documentation

### Lower Priority
- [ ] Performance optimization
- [ ] Additional layer types
- [ ] Extended examples

## Code Standards

- Follow PEP 8
- Use type hints
- Write docstrings
- Include tests
- Keep functions small and focused

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## Questions?

Open an issue or discussion.

Welcome to the revolution!