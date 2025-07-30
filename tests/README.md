# ChatOS Tests

This folder contains all ChatOS testing infrastructure.

## Structure

```
tests/
├── monitor_fixed_simple.py     # Main performance monitor
├── test_suite.py               # Comprehensive test suite  
├── test_basics.py              # Basic functionality tests
├── test_config.json            # Test configuration
├── requirements-test.txt       # Test dependencies
└── monitors/                   # Performance monitor variants
    ├── performance_monitor_OLD.py  # Original (broken) version
    ├── simple_monitor.py           # Basic working version
    └── working_monitor.py          # Another working version
```

## Running Tests

### From ChatOS root directory:

```bash
# Run all tests (organized)
python run_tests_organized.py

# Run individual tests
python tests/monitor_fixed_simple.py
python tests/test_suite.py  
python tests/test_basics.py
```

### Performance Monitoring

The main performance monitor is `monitor_fixed_simple.py`:
- Tracks ChatOS process memory (not system memory)
- Realistic thresholds (200MB for ChatOS processes)
- No false alerts about system memory
- Proper process detection

### Test Configuration

Edit `test_config.json` to adjust:
- Performance thresholds
- Test timeouts
- Mock settings for CI/CD

## Key Metrics

- **ChatOS Memory**: <200MB (realistic for Python processes)
- **System Memory**: <85% (percentage-based)
- **Local Model Usage**: >50%
- **Tool Success Rate**: >85%
- **Response Time**: <5 seconds
