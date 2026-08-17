BigBits
=======

BigBits is a Rust 2021 library for high-performance, multiple-precision
integer arithmetic. It combines convenient signed and unsigned integer types
with optimized low-level arithmetic over 64-bit limbs.

The library supports both dynamically sized integers and fixed-capacity,
array-backed integers, allowing users to choose between flexible storage and
a predetermined memory footprint.


Development Status
------------------

BigBits is currently under active development.

The library is not recommended for production use. Its public API, arithmetic
behavior, internal algorithms, and platform support may change between
revisions. BigBits has not undergone a formal security or correctness audit.


Features
--------

- Arbitrary-precision unsigned integers
- Arbitrary-precision signed integers
- Fixed-capacity unsigned integers
- Fixed-capacity signed integers
- Addition and subtraction
- Multiplication and optimized squaring
- Division and remainder
- Integer exponentiation
- Integer logarithms
- Left and right bit shifts
- Signed negation and absolute-value operations
- Comparisons and ordering
- Decimal string parsing and formatting
- Conversions to and from Rust primitive integers
- Operations between BigBits values and primitive integer types
- Low-level APIs for direct limb-buffer arithmetic
- Architecture-specific optimizations for x86-64 and AArch64
- Parallel processing for sufficiently large arithmetic operations


Number Types
------------

UBitInt

An unsigned arbitrary-precision integer backed by a dynamically sized
`Vec<u64>`. Its storage grows as needed to represent larger values.


BitInt

A signed arbitrary-precision integer backed by a dynamically sized magnitude
and a separate sign.


UBitIntStatic<N>

An unsigned fixed-capacity integer backed by `[u64; N]`. The type can store a
magnitude of up to `N * 64` bits without dynamically expanding its primary
storage.


BitIntStatic<N>

A signed fixed-capacity integer backed by an `N`-limb magnitude and a separate
sign.


Basic Usage
-----------

    use big_bits::bit_nums::traits::{DivRem, PowI, Sqr};
    use big_bits::bit_nums::ubitint::UBitInt;

    fn main() {
        let a: UBitInt = "123456789012345678901234567890"
            .parse()
            .expect("valid integer");

        let b = UBitInt::from(42_u64);

        let sum = &a + &b;
        let product = &a * &b;
        let square = a.sqr();
        let power = b.powi(10);

        let (quotient, remainder) = (&product).div_rem(&a);

        println!("sum       = {sum}");
        println!("product   = {product}");
        println!("square    = {square}");
        println!("42^10     = {power}");
        println!("quotient  = {quotient}");
        println!("remainder = {remainder}");
    }


Fixed-Capacity Integers
-----------------------

Fixed-capacity types use a compile-time limb count:

    use big_bits::bit_nums::ubitint_static::UBitIntStatic;

    type U256 = UBitIntStatic<4>;

    let a = U256::from(12_u64);
    let b = U256::from(34_u64);
    let product = a * b;

Because the capacity cannot grow, the result of an operation must fit within
the selected number of limbs. Fixed-capacity arithmetic should not be treated
as checked arithmetic unless the specific operation documents that behavior.


Arithmetic Implementation
-------------------------

BigBits represents integer magnitudes as little-endian sequences of `u64`
limbs. The limb at index zero contains the least significant 64 bits.

Multiplication adaptively selects an algorithm according to the sizes and
shapes of its operands. Available implementations include:

- Single-limb multiplication
- Double-limb multiplication
- Schoolbook multiplication
- Karatsuba multiplication
- FFT-based multiplication
- Number Theoretic Transform multiplication

The dynamic integer path can use FFT or NTT multiplication for large values.
The fixed-capacity path uses algorithms designed around caller-selected array
storage. Large NTT operations can use parallel execution.

Specialized low-level multiplication functions are also available for
squaring and for computing selected portions of a product.

Division similarly selects between multiple algorithms, including:

- Knuth division
- Burnikel-Ziegler division
- Newton-Raphson reciprocal refinement

The division API supports quotient-only operations, quotient-and-remainder
operations, and reciprocal approximations.


Primitive Integer Interoperability
----------------------------------

BigBits values can be created from common Rust primitive integers. Supported
operations also accept primitive values in many common expressions.

Examples include:

    let a = UBitInt::from(100_u64);
    let b = &a + 25_u64;
    let c = &b * 10_u128;

Values that fit within the destination type can be converted back using
`TryFrom`:

    let value = UBitInt::from(500_u64);
    let primitive = u64::try_from(value).expect("value fits in u64");


String Conversion
-----------------

The integer types support decimal parsing through `FromStr`:

    let value: UBitInt = "999999999999999999999999"
        .parse()
        .expect("valid unsigned integer");

Signed values may be parsed using `BitInt`:

    use big_bits::bit_nums::bitint::BitInt;

    let value: BitInt = "-12345678901234567890"
        .parse()
        .expect("valid signed integer");

All four public integer types implement `Display` for decimal output.


Low-Level API
-------------

The `utils` module exposes lower-level functions that operate directly on
limb slices. These functions provide access to the arithmetic routines used
by the higher-level number types.

Low-level values use little-endian `u64` buffers. Many of these functions
expect normalized operands, correctly sized output buffers, and other
algorithm-specific preconditions. They are intended for users who need
direct control over allocation and arithmetic buffers.


Platform Support
----------------

BigBits currently targets:

- x86-64
- AArch64 / ARM64

Performance-critical primitive operations use architecture-specific inline
assembly. Other processor architectures are not currently supported.


Repository
----------

Source code is available at:

https://github.com/AidanRS5772/BigBits

The current `master` branch represents the public state of the project.


Safety Notice
-------------

BigBits contains unsafe Rust and architecture-specific inline assembly in
performance-critical routines. Although the project includes tests for
arithmetic operations and boundary conditions, it remains development
software.

Do not rely on BigBits for cryptography, financial systems, safety-critical
applications, or other production workloads where incorrect arithmetic could
cause harm.
