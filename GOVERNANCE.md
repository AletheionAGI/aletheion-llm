# Governance

This document outlines the governance structure and decision-making processes for the Aletheion LLM project.

## Project Vision

Aletheion LLM aims to reduce overconfidence in Large Language Models through architectural uncertainty quantification using Q1 (aleatoric) and Q2 (epistemic) gates, with a derived Height coordinate for calibration and safety.

## Governance Model

Aletheion LLM follows a **benevolent dictator** model with community input, transitioning toward a **consensus-based governance** as the project matures.

## Roles and Responsibilities

### Project Lead

**Current Lead**: Felipe M. Muniz

**Responsibilities**:
* Final decision-making authority on project direction
* Architecture and design decisions
* Release management and versioning
* Repository access and permissions
* Strategic planning and roadmap
* Community representation

### Core Contributors

Core contributors are individuals who have made substantial, ongoing contributions to the project.

**Responsibilities**:
* Code review and approval
* Architecture discussions
* Mentoring new contributors
* Issue triage and management
* Documentation maintenance

**Becoming a Core Contributor**:
* Sustained high-quality contributions over 3+ months
* Deep understanding of project architecture
* Demonstrated commitment to code quality and testing
* Active participation in design discussions
* Nomination by existing core contributor + approval by project lead

**Current Core Contributors**:
* Felipe M. Muniz (Project Lead)
* (Open for nominations)

### Contributors

Anyone who contributes code, documentation, bug reports, or other improvements to the project.

**How to Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

### Users

Anyone who uses Aletheion LLM in their projects, research, or products.

**Feedback Welcome**: File issues, participate in discussions, share use cases

## Decision-Making Process

### Types of Decisions

#### 1. Routine Decisions
**Examples**: Bug fixes, documentation updates, minor refactoring

**Process**:
* Submit pull request
* Review by any core contributor
* Merge upon approval

#### 2. Minor Technical Decisions
**Examples**: New features, API additions, dependency updates

**Process**:
* Create issue or RFC (Request for Comments)
* Discussion period (minimum 3 days)
* Core contributor review
* Project lead approval for merging

#### 3. Major Technical Decisions
**Examples**: Breaking API changes, architecture changes, new levels/gates

**Process**:
* Create detailed RFC with:
  * Problem statement
  * Proposed solution
  * Alternatives considered
  * Impact analysis
  * Migration path (if applicable)
* Community discussion period (minimum 1 week)
* Core contributor consensus
* Project lead final decision

#### 4. Governance Decisions
**Examples**: New core contributors, governance changes, licensing

**Process**:
* Proposal by core contributor or project lead
* Discussion with all core contributors
* Consensus required
* Project lead final decision

### Consensus Building

* **Soft consensus**: No strong objections
* **Hard consensus**: Active agreement from all core contributors
* **Disagreement resolution**: Project lead makes final decision

### Transparency

All significant decisions are documented in:
* GitHub issues and pull requests
* `CHANGELOG.md` for user-facing changes
* Meeting notes (if applicable)
* Design documents in `docs/`

## Communication Channels

### Official Channels

* **GitHub Issues**: Bug reports, feature requests, technical discussions
* **Pull Requests**: Code review and technical implementation discussions
* **GitHub Discussions**: General questions, ideas, announcements (when enabled)
* **Email**: contact@alethea.tech (project lead)

### Discussion Guidelines

* Be respectful and professional (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md))
* Stay on topic
* Provide constructive feedback
* Support claims with evidence
* Consider diverse perspectives

## Pull Request Review Process

### Review Requirements

| Change Type | Approvals Required |
|-------------|-------------------|
| Documentation, typos | 1 contributor |
| Bug fixes | 1 core contributor |
| New features | 1 core contributor + project lead review |
| Breaking changes | All core contributors + project lead approval |
| Security fixes | Project lead approval |

### Review Criteria

Reviewers should verify:
* **Functionality**: Does it work as intended?
* **Tests**: Are there adequate tests with good coverage?
* **Documentation**: Is it properly documented?
* **Code Quality**: Does it follow style guidelines?
* **Performance**: Are there performance implications?
* **Security**: Are there security considerations?
* **Breaking Changes**: Is the migration path clear?

### Timeline Expectations

* **Initial review**: Within 1 week for most PRs
* **Follow-up**: Within 3 days for revisions
* **Stale PRs**: Closed after 60 days of inactivity (with notice)

## Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/) (SemVer):

* **MAJOR**: Breaking API changes
* **MINOR**: New features, backward-compatible
* **PATCH**: Bug fixes, backward-compatible

### Release Cycle

* **Patch releases**: As needed for critical bugs
* **Minor releases**: Every 1-3 months
* **Major releases**: When breaking changes are necessary

### Release Approval

* Project lead authorizes all releases
* Core contributors review release notes
* Security audit for major releases (when resources available)

### Release Checklist

See [scripts/make_release.sh](scripts/make_release.sh) for automated process:

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Git tag created
- [ ] PyPI package published
- [ ] GitHub release created
- [ ] Announcement posted

## Code of Conduct Enforcement

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for the full code.

**Enforcement Authority**: Project lead

**Process**:
1. Report to contact@alethea.tech
2. Investigation by project lead
3. Decision and action within 7 days
4. Right to appeal

## Intellectual Property

### Copyright

* All contributors retain copyright to their contributions
* Contributions licensed under AGPL-3.0-or-later
* Copyright notices should not be removed

### Contributor License

By contributing, you agree that your contributions will be licensed under the same license as the project (AGPL-3.0-or-later).

### Commercial Licensing

For commercial licensing options, see [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) or contact contact@alethea.tech.

## Amendments to Governance

This governance document may be amended by:

1. Proposal by core contributor or project lead
2. Discussion period (minimum 2 weeks)
3. Consensus of all core contributors
4. Project lead approval
5. Update to this document with change history

## Change History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-01-07 | 1.0 | Initial governance document | Felipe M. Muniz |

## Acknowledgments

This governance model is inspired by:
* [Python's PEP 8000](https://peps.python.org/pep-8000/)
* [Django's Governance](https://www.djangoproject.com/foundation/teams/)
* [NumPy's Governance](https://numpy.org/devdocs/dev/governance/governance.html)
* [Open Source Guides](https://opensource.guide/leadership-and-governance/)

---

**Questions?** Contact contact@alethea.tech or open an issue for discussion.
