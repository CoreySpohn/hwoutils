# Changelog

All notable changes to this project will be documented in this file.

## [1.3.0](https://github.com/CoreySpohn/hwoutils/compare/v1.2.0...v1.3.0) (2026-02-26)


### Features

* Add gaussian FWHM constant ([f3690c0](https://github.com/CoreySpohn/hwoutils/commit/f3690c0f2a7bde4fb550a8d1c2b526c66bb6369c))

## [1.2.0](https://github.com/CoreySpohn/hwoutils/compare/v1.1.0...v1.2.0) (2026-02-25)


### Features

* Adding jax configuration utility functions ([a20e203](https://github.com/CoreySpohn/hwoutils/commit/a20e203abc8fd3f1fc9c26e586023da9c215acf9))

## [1.1.0](https://github.com/CoreySpohn/hwoutils/compare/v1.0.0...v1.1.0) (2026-02-23)


### Features

* Adding basic tests ([9010750](https://github.com/CoreySpohn/hwoutils/commit/901075009e812c16c90eacd6e257ef308164d9d4))


### Bug Fixes

* Fix the calculation in map_coordinates to compute the full product for interpolation ([b453aeb](https://github.com/CoreySpohn/hwoutils/commit/b453aeb4905199802dcb931bb36b1b88a9d0d456))

## 1.0.0 (2026-02-21)


### Features

* Initial migration of repeated code ([9150db9](https://github.com/CoreySpohn/hwoutils/commit/9150db9bac1368570d4177504377862668e5c164))

## [Unreleased]

### Added

- `constants` module with consolidated physical constants from orbix and coronagraphoto
- `conversions` module with JAX-native unit conversion functions
- `map_coordinates` module with cubic spline interpolation (from JAX PR #14218)
- `transforms` module with `resample_flux`, `ccw_rotation_matrix`, and `shift_image`
