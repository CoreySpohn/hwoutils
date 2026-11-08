# Changelog

All notable changes to this project will be documented in this file.

## [1.14.0](https://github.com/CoreySpohn/hwoutils/compare/v1.13.0...v1.14.0) (2026-11-08)


### Features

* **testbudget:** charge fixture setup to its users, module mode, tier reasons, session JSON ([eec97f2](https://github.com/CoreySpohn/hwoutils/commit/eec97f2be6441b834fe749afd780f928d6cc53ca))


### Bug Fixes

* **testbudget:** charge the fixtures a test resolved, count cache loads in module program budgets ([85f0d91](https://github.com/CoreySpohn/hwoutils/commit/85f0d91a6bba1e721c3b56c9ef63cb2a49ee9da8))
* **testbudget:** link fixtures a setup looks up while cached to the fixture that looked them up ([761c9a1](https://github.com/CoreySpohn/hwoutils/commit/761c9a1a02dd3ca922aabc7776f1eccaf4573db9))
* **testbudget:** share fixtures set up inside other fixtures, snapshot users at call end, count loads when they close ([1dee427](https://github.com/CoreySpohn/hwoutils/commit/1dee4273f59c17f64abc8039d0bc2b28a8f5e9a1))

## [1.13.0](https://github.com/CoreySpohn/hwoutils/compare/v1.12.0...v1.13.0) (2026-11-08)


### Features

* **testbudget:** record compile seconds and add module_compile_seconds budgets ([47a3fb8](https://github.com/CoreySpohn/hwoutils/commit/47a3fb8bf50f5461d27861010f1fb98c60fe73ba))


### Bug Fixes

* **testbudget:** key modules by relative path, judge revisits cumulatively, reset hits globally ([3c1bcfd](https://github.com/CoreySpohn/hwoutils/commit/3c1bcfd88462b2b72256155da6396a68d0da7ead))
* **testbudget:** reset pending cache hits per thread and cover module key precedence ([efdcddc](https://github.com/CoreySpohn/hwoutils/commit/efdcddcf1028862f50bf37456ba1bab069ad8c88))
* **testbudget:** sum module budgets across xdist workers and pair nested cache hits ([ddecd84](https://github.com/CoreySpohn/hwoutils/commit/ddecd84462601de14228ff58e7c871d624a155dd))

## [1.12.0](https://github.com/CoreySpohn/hwoutils/compare/v1.11.0...v1.12.0) (2026-11-07)


### Features

* **conversions:** MJy/sr and Jy/arcsec^2 surface-brightness conversions ([f291d82](https://github.com/CoreySpohn/hwoutils/commit/f291d827c3ff7384e02b0ce3a1e8c6c470f30b0f))

## [1.11.0](https://github.com/CoreySpohn/hwoutils/compare/v1.10.0...v1.11.0) (2026-11-05)


### Features

* **constants:** add IAU 2015 nominal solar radius Rsun2m and Rsun2AU ([602295d](https://github.com/CoreySpohn/hwoutils/commit/602295d5f064eca5b74cff70f244d0c848cf0853))
* **testbudget:** pytest guard for per-test time, memory ceiling, compile counts and trend file ([7303554](https://github.com/CoreySpohn/hwoutils/commit/73035540db0e4ed7c6066b9c2cb17c1ae3799b7e))


### Bug Fixes

* **hwoutils:** preserve rotation center tuple type ([683aa52](https://github.com/CoreySpohn/hwoutils/commit/683aa5221a04775956076cc4b96b59a38a5e2bc7))

## [1.10.0](https://github.com/CoreySpohn/hwoutils/compare/v1.9.0...v1.10.0) (2026-09-14)


### Features

* **transforms:** add conservative rebin_flux and correct rotation/interpolation edge cases ([c106f3e](https://github.com/CoreySpohn/hwoutils/commit/c106f3e24bf39f22af1aea0f43503f4a0dc8a260))

## [1.9.0](https://github.com/CoreySpohn/hwoutils/compare/v1.8.0...v1.9.0) (2026-07-10)


### Features

* **conversions:** AB mag &lt;-&gt; Jy helpers (astropy-exact zero point; canonical home for orbix/skyscapes/jaxedith) ([b11640e](https://github.com/CoreySpohn/hwoutils/commit/b11640ebef5078317d058421018a7173980a3253))

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
