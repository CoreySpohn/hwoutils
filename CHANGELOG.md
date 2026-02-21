# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0 (2026-02-21)


### Features

* Initial migration of repeated code ([9150db9](https://github.com/CoreySpohn/hwoutils/commit/9150db9bac1368570d4177504377862668e5c164))

## [Unreleased]

### Added

- `constants` module with consolidated physical constants from orbix and coronagraphoto
- `conversions` module with JAX-native unit conversion functions
- `map_coordinates` module with cubic spline interpolation (from JAX PR #14218)
- `transforms` module with `resample_flux`, `ccw_rotation_matrix`, and `shift_image`
