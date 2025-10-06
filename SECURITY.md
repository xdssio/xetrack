# Security Policy

## Supported Versions

We actively support the following versions of Xetrack with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| 0.3.x   | :x:                |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Xetrack, please help us by reporting it responsibly.

### How to Report

**Please do NOT create a public GitHub issue for security vulnerabilities.**

Instead, please report security vulnerabilities by:

1. **Email**: Send a detailed report to [jonathan@xdss.io](mailto:jonathan@xdss.io)
2. **Subject Line**: Include "XETRACK SECURITY" in the subject line
3. **Include**: 
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Any suggested fixes (if you have them)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Investigation**: We will investigate the issue and determine its severity
- **Timeline**: We aim to provide an initial assessment within 5 business days
- **Resolution**: For confirmed vulnerabilities, we will work on a fix and coordinate disclosure

### Security Considerations

#### Database Security

Xetrack works with local database files (SQLite/DuckDB). Users should be aware of:

- **File Permissions**: Database files may contain sensitive experiment data
- **Network Access**: Xetrack is designed for local use and doesn't include network security features
- **Asset Storage**: Pickled objects in asset storage could potentially execute code when loaded

#### CLI Security

- **Command Injection**: Be cautious with user-provided SQL queries via the CLI
- **File Access**: CLI commands can read/write database files with user permissions

#### Python Security

- **Pickle Deserialization**: Asset functionality uses cloudpickle, which can execute arbitrary code
- **Dependencies**: Keep dependencies updated to avoid known vulnerabilities

### Best Practices for Users

1. **Keep Updated**: Always use the latest supported version
2. **File Permissions**: Secure your database files with appropriate permissions
3. **Trusted Assets**: Only load assets from trusted sources
4. **Environment**: Use Xetrack in controlled environments for sensitive data

### Security Features

- **No Network Components**: Xetrack operates locally, reducing attack surface
- **Optional Asset Loading**: Asset functionality is optional and can be disabled
- **Read-only Reader**: The Reader class provides read-only access to data

## Responsible Disclosure

We follow responsible disclosure practices:

1. We will work with you to understand and validate the reported vulnerability
2. We will develop and test a fix
3. We will coordinate the timing of the public disclosure
4. We will credit you for the discovery (unless you prefer to remain anonymous)

## Questions

If you have questions about this security policy, please contact [jonathan@xdss.io](mailto:jonathan@xdss.io).

---

Thank you for helping keep Xetrack secure! 🔒