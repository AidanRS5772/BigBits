<h2>Big Bits</h2>

<h6>In Progress</h6>

**Big Bits** is an all-in-one Big Integer, Big Fraction, Arbitrary Precision Floating point library written in Rust. It has the lofty goals of the following:

  1. **Speed** - The first and foremost priority
  2. **Easy** - Make it as easy as possible for the proammamer or scientist to get what they need done
  3. **Versatile** - Allowing for a wide range of use cases and applications

The Big Integer implementations come in signed and unsigned variants (UBitInt, UBitIntStatic, BitInt, BitIntStatic). The floating point implimentations also come in stack and heap defined versions (BitFloat, BitFloatStatic). The implimimentations allow for the programmer to make the descision for how exactly they would like things to function and what trade offs they want.

The motivation for writing this library was encountering my own frustrations with the ecosystem of Big Math libraries across languages. The design of the API and algorithmic choices made here are intended to alleviate some of those frustrations. The API is not stable as of now. I do not recommend using this in production builds yet.

<h6> Features in Progress ...</h6>

  1. Full type optimized type compatibility
  2. Speed optimizations
  3. Thorogh testing and benchmarking

<h6> Features Coming Soon </h6>

  1. Compile time compatability
  2. Full trigonometric and special function suite
  3. Cross-language wrappers




