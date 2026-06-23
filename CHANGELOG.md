# Changelog

All notable changes to this project will be documented in this file.

## [1.8.0](https://github.com/CoreySpohn/hwoutils/compare/v1.7.0...v1.8.0) (2026-06-23)


### Miscellaneous Chores

* release 1.8.0 ([f831550](https://github.com/CoreySpohn/hwoutils/commit/f831550e6e786e655e405b51050e5c538f08f940))

## [1.7.0](https://github.com/CoreySpohn/hwoutils/compare/v1.6.1...v1.7.0) (2026-06-23)


### Features

* Add rotate_image fft function ([0a9c48e](https://github.com/CoreySpohn/hwoutils/commit/0a9c48e572084fed8998244e50c692d9778d9c7f))
* Add Zenodo info ([542caa0](https://github.com/CoreySpohn/hwoutils/commit/542caa058e273532459db5d074d179e666c75700))

## [1.6.1](https://github.com/CoreySpohn/hwoutils/compare/v1.6.0...v1.6.1) (2026-05-25)


### Bug Fixes

* Update naming to match conventions ([75ee546](https://github.com/CoreySpohn/hwoutils/commit/75ee5468287d0dccb3ae95985e541aa68f6a35a6))

## [1.6.0](https://github.com/CoreySpohn/hwoutils/compare/v1.5.1...v1.6.0) (2026-05-22)


### Features

* Albedo conversion ([4828f28](https://github.com/CoreySpohn/hwoutils/commit/4828f28febae437ffbcd0336918a8dc5ef22322d))
* New basic conversions ([115dc49](https://github.com/CoreySpohn/hwoutils/commit/115dc4917eb446471ea687a6d59cbba4ca144f53))

## [1.5.1](https://github.com/CoreySpohn/hwoutils/compare/v1.5.0...v1.5.1) (2026-04-23)


### Bug Fixes

* **interp:** use Keys cubic convolution for order=3 ([b673b02](https://github.com/CoreySpohn/hwoutils/commit/b673b02d575582a5c47214a1254d5b19a301b83c))

## [1.5.0](https://github.com/CoreySpohn/hwoutils/compare/v1.4.0...v1.5.0) (2026-04-13)


### Features

* Add simple area conversions ([0925c89](https://github.com/CoreySpohn/hwoutils/commit/0925c89c17d4119f451d2e63aa27224826411a48))
* add snapshot module for reproducible workspace tracking ([03699dd](https://github.com/CoreySpohn/hwoutils/commit/03699dd11257e47cd55876cac6cee05d3fee6cd2))

## [1.4.0](https://github.com/CoreySpohn/hwoutils/compare/v1.3.1...v1.4.0) (2026-03-04)


### Features

* Add fft utility functions ([0aa89ac](https://github.com/CoreySpohn/hwoutils/commit/0aa89acdcc67026b182290d999988d3134ec5ab9))


### Bug Fixes

* Remove hardcoded dtype in fft shift functions ([4d90db2](https://github.com/CoreySpohn/hwoutils/commit/4d90db2d6fcb0546857d20fd51694572e0057f15))

## [1.3.1](https://github.com/CoreySpohn/hwoutils/compare/v1.3.0...v1.3.1) (2026-02-26)


### Bug Fixes

* Set minimum jax version ([6e5193d](https://github.com/CoreySpohn/hwoutils/commit/6e5193d858591aa5e5f5a1d85146f1fbd4882044))

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
