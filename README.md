High Performance Big Integer and Big Float Library built in Rust. Using inline assembly routines that support both Arm 64-bit archetectures and x86 64-bit archetectures I was able to make substanitial improvments to the current eco-system of Large Numerical Libraries. This is done by leveraging the incredible ability of the Rust Copiler to simoultaneously optimize high level code, integrate low level systems, and at zero cost abstract away operations. This leads to a blazingly fast and extreamly simple to use library that is incredible verbose in its operations while maintaining low level control like if you would like stack or heap allocated numbers. All of the operations included in the library have been optimized to the greatest extent for performance using the most up to date research in theoretical computer science and best practices of memory mangment in rust.
