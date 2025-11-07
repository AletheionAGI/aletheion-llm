# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for
receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

The Aletheion LLM team and community take security bugs seriously. We appreciate
your efforts to responsibly disclose your findings, and will make every effort to
acknowledge your contributions.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**security@alethea.tech**

If you prefer, you can also use our contact email:

**contact@alethea.tech**

Please include the following information in your report:

* Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

This information will help us triage your report more quickly.

### What to Expect

After you have submitted a vulnerability report, you should expect:

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within
   72 hours.

2. **Investigation**: We will investigate the issue and determine its impact and severity.

3. **Updates**: We will keep you informed about our progress as we work to address the
   vulnerability.

4. **Resolution Timeline**: We aim to provide an initial assessment within 7 days and a
   full fix within 90 days, depending on complexity.

5. **Public Disclosure**: We follow coordinated disclosure practices. We will work with
   you to determine an appropriate disclosure timeline that protects users while giving
   you credit for the discovery.

## Security Update Process

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported releases
4. Release new security fix versions as soon as possible

## Security Best Practices for Users

If you're using Aletheion LLM in production:

1. **Keep Updated**: Always use the latest stable version
2. **Monitor Announcements**: Watch our repository for security announcements
3. **Dependency Scanning**: Regularly scan dependencies for known vulnerabilities
4. **Principle of Least Privilege**: Run the application with minimal required permissions
5. **Input Validation**: Always validate and sanitize inputs when using the library
6. **Secure Configuration**: Follow secure configuration guidelines in our documentation

## Known Security Considerations

### Model Security

* **Training Data**: Be aware that models may have been trained on sensitive data
* **Adversarial Inputs**: Models may be vulnerable to adversarial examples
* **Prompt Injection**: When using LLMs, be cautious of prompt injection attacks
* **Output Validation**: Always validate model outputs before use in production systems

### Dependency Security

We actively monitor our dependencies for security vulnerabilities. Our CI/CD pipeline
includes automated security scans. However, users should also:

* Regularly update dependencies
* Use tools like `pip-audit` or `safety` to scan for vulnerabilities
* Review our `requirements.txt` and `pyproject.toml` for security implications

## Responsible Disclosure

We kindly ask that you:

* Give us reasonable time to investigate and fix the issue before public disclosure
* Make a good faith effort to avoid privacy violations, data destruction, and service
  interruption
* Not access or modify data that doesn't belong to you
* Not perform any attack that could harm the reliability or integrity of our services

## Recognition

We believe in recognizing security researchers who help us maintain the security of
Aletheion LLM. With your permission, we will:

* Acknowledge your responsible disclosure in our release notes
* List you in our security acknowledgments (unless you prefer to remain anonymous)

## Security Contacts

* **Security Email**: security@alethea.tech
* **General Contact**: contact@alethea.tech
* **Project Maintainer**: Felipe M. Muniz

## Additional Resources

* [OWASP Top 10](https://owasp.org/www-project-top-ten/)
* [CWE Top 25](https://cwe.mitre.org/top25/)
* [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

## Policy Updates

This security policy may be updated from time to time. Please check back regularly
for any changes.

---

**Last Updated**: 2025-01-07
