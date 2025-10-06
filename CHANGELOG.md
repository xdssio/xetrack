# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow for automated testing
- CONTRIBUTING.md with comprehensive development guidelines
- SECURITY.md with security policy and vulnerability reporting
- MIT License file for open source compliance

### Changed
- Updated license from BSD-3-Clause to MIT

### Fixed
- Version consistency between pyproject.toml and package metadata

## [0.4.3] - 2023-10-09

### Added
- Enhanced DuckDB support with improved table name handling
- Copy functionality between databases with asset preservation
- Better error handling for database operations

### Changed
- Improved engine abstraction layer
- Better multiprocess safety for concurrent operations

### Fixed
- DuckDB table naming issues with quoted identifiers
- Asset deduplication edge cases

## [0.3.4] - 2023-08-22

### Added
- Asset management with deduplication using xxhash
- CLI interface using Typer framework
- Support for both SQLite and DuckDB engines
- System and network parameter tracking
- Git commit hash tracking

### Changed
- Refactored engine architecture for better abstraction
- Improved logging integration with loguru

### Fixed
- Memory usage optimization for large datasets
- CLI command consistency across engines

## [0.3.0] - 2023-08-15

### Added
- Core tracking functionality
- Dynamic schema management
- Pandas DataFrame integration
- Function tracking and wrapping
- Basic CLI commands

### Changed
- Major refactoring of core architecture
- Improved type hints throughout codebase

### Fixed
- SQLite type coercion issues
- Multiprocessing compatibility

## [0.2.0] - 2023-07-01

### Added
- Basic experiment tracking
- SQLite backend support
- Simple parameter management

### Fixed
- Initial bug fixes and stability improvements

## [0.1.0] - 2023-06-15

### Added
- Initial release
- Basic tracking functionality
- SQLite database support

---

## How to Update

To update to the latest version:

```bash
pip install --upgrade xetrack

# Or with specific extras
pip install --upgrade "xetrack[duckdb,assets]"
```

## Migration Notes

### From 0.3.x to 0.4.x
- No breaking changes for basic functionality
- DuckDB users may benefit from improved table handling
- Asset management is more robust

### From 0.2.x to 0.3.x
- Major API changes - review documentation
- Engine abstraction introduced
- CLI commands added

### From 0.1.x to 0.2.x
- Database schema changes may require recreation
- Parameter handling improved

---

For more details on any version, see the [GitHub releases](https://github.com/xdssio/xetrack/releases).