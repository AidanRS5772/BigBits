pub mod bitfloat;
pub mod bitfrac;
pub mod bitint;
pub mod bitint_static;
pub mod ubitint;
pub mod ubitint_static;

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod ubi_tests {

        use crate::ubitint::UBitInt;

        #[test]
        fn ubi_eq_test() {
            let (a1, a2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
            );
            assert_eq!(a1 == a2, true);

            let (b1, b2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, 0, usize::MAX]),
            );
            assert_eq!(b1 == b2, false);

            let (c1, c2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
            );
            assert_eq!(c1 == c2, false);

            let (d1, d2) = (UBitInt::make(vec![usize::MAX, usize::MAX]), u128::MAX);
            assert_eq!(d1 == d2, true);

            let (e1, e2) = (UBitInt::make(vec![usize::MAX, usize::MAX]), u128::MAX - 1);
            assert_eq!(e1 == e2, false);

            let (f1, f2) = (UBitInt::make(vec![10]), 10_u128);
            assert_eq!(f1 == f2, true);

            let (g1, g2) = (UBitInt::make(vec![10]), 10_u64);
            assert_eq!(g1 == g2, true);

            let (h1, h2) = (UBitInt::make(vec![10]), 10_u32);
            assert_eq!(h1 == h2, true);
        }

        #[test]
        fn ubi_cmp_test() {
            use std::cmp::Ordering::*;

            let (a1, a2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
            );
            assert_eq!(a1.cmp(&a2), Equal);
            assert_eq!(a2.cmp(&a1), Equal);

            let (b1, b2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, 0, usize::MAX]),
            );
            assert_eq!(b1.cmp(&b2), Greater);
            assert_eq!(b2.cmp(&b1), Less);

            let (c1, c2) = (
                UBitInt::make(vec![usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
            );
            assert_eq!(c1.cmp(&c2), Less);
            assert_eq!(c2.cmp(&c1), Greater);
        }

        #[test]
        fn ubi_add_test() {
            let (a1, a2, resa) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![1]),
                UBitInt::make(vec![0, 0, 0, 1]),
            );
            assert_eq!(&a1 + &a2, resa);
            assert_eq!(a2 + a1, resa);

            let (b1, b2, resb) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX - 1, usize::MAX, usize::MAX, 1]),
            );
            assert_eq!(&b1 + &b2, resb);
            assert_eq!(b2 + b1, resb);

            let (c1, c2, resc) = (
                UBitInt::make(vec![10]),
                UBitInt::make(vec![10]),
                UBitInt::make(vec![20]),
            );
            assert_eq!(&c1 + &c2, resc);
            assert_eq!(c2 + c1, resc);
        }

        #[test]
        fn ubi_sub_test() {
            let (a1, a2, resa) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, 1]),
                UBitInt::make(vec![usize::MAX, usize::MAX]),
                UBitInt::make(vec![0, 0, 1]),
            );
            assert_eq!(a1 - a2, resa);

            let (b1, b2, resb) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![]),
            );
            assert_eq!(b1 - b2, resb);

            let (c1, c2, resc) = (
                UBitInt::make(vec![10]),
                UBitInt::make(vec![8]),
                UBitInt::make(vec![2]),
            );
            assert_eq!(c1 - c2, resc);

            let (d1, d2, resd) = (
                UBitInt::make(vec![usize::MAX, usize::MAX, usize::MAX]),
                UBitInt::make(vec![0, usize::MAX, usize::MAX]),
                UBitInt::make(vec![usize::MAX]),
            );
            assert_eq!(d1 - d2, resd);
        }
    }

    #[cfg(test)]
    mod bf_test {

        use crate::bitfloat::BitFloat;

        #[test]
        fn bf_cmp_test() {
            use std::cmp::Ordering::*;

            let (a1, a2) = (
                BitFloat::make(vec![1, 2, 3], 1, false),
                BitFloat::make(vec![1, 2, 3], 1, false),
            );

            assert_eq!(a1.cmp(&a2), Equal);

            let (b1, b2) = (
                BitFloat::make(vec![1, 2, 3], 0, false),
                BitFloat::make(vec![1, 2, 3], 1, false),
            );

            assert_eq!(b1.cmp(&b2), Less);

            let (c1, c2) = (
                BitFloat::make(vec![1, 2, 3], 0, false),
                BitFloat::make(vec![1, 2, 3], 1, true),
            );

            assert_eq!(c1.cmp(&c2), Greater);

            let (d1, d2) = (
                BitFloat::make(vec![2, 2, 3], 1, false),
                BitFloat::make(vec![1, 2, 3], 1, false),
            );

            assert_eq!(d1.cmp(&d2), Greater);
        }

        #[test]
        fn bf_add_bench() {
            let (a1, a2, resa) = (
                BitFloat::make(vec![1], 0, false),
                BitFloat::make(vec![1], 1, false),
                BitFloat::make(vec![1, 1], 1, false),
            );

            assert_eq!(a1 + a2, resa);

            let (b1, b2, resb) = (
                BitFloat::make(vec![1], 0, true),
                BitFloat::make(vec![1], 1, false),
                BitFloat::make(vec![usize::MAX], 0, false),
            );

            assert_eq!(b1 + b2, resb);

            let (c1, c2, resc) = (
                BitFloat::make(vec![1], 0, false),
                BitFloat::make(vec![usize::MAX], -1, true),
                BitFloat::make(vec![1], -1, false),
            );

            assert_eq!(c1 + c2, resc);

            let (d1, d2, resd) = (
                BitFloat::make(vec![usize::MAX], 0, false),
                BitFloat::make(vec![usize::MAX, usize::MAX], 0, true),
                BitFloat::make(vec![usize::MAX], -1, true),
            );

            assert_eq!(d1 + d2, resd);

            let (e1, e2, rese) = (
                BitFloat::make(vec![usize::MAX], 0, true),
                BitFloat::make(vec![usize::MAX, usize::MAX], 0, false),
                BitFloat::make(vec![usize::MAX], -1, false),
            );

            assert_eq!(e1 + e2, rese);
        }

        #[cfg(test)]
        mod bf_test {

            use crate::bitfloat::BitFloat;

            #[test]
            fn bf_cmp_test() {
                use std::cmp::Ordering::*;

                let (a1, a2) = (
                    BitFloat::make(vec![1, 2, 3], 1, false),
                    BitFloat::make(vec![1, 2, 3], 1, false),
                );

                assert_eq!(a1.cmp(&a2), Equal);

                let (b1, b2) = (
                    BitFloat::make(vec![1, 2, 3], 0, false),
                    BitFloat::make(vec![1, 2, 3], 1, false),
                );

                assert_eq!(b1.cmp(&b2), Less);

                let (c1, c2) = (
                    BitFloat::make(vec![1, 2, 3], 0, false),
                    BitFloat::make(vec![1, 2, 3], 1, true),
                );

                assert_eq!(c1.cmp(&c2), Greater);

                let (d1, d2) = (
                    BitFloat::make(vec![2, 2, 3], 1, false),
                    BitFloat::make(vec![1, 2, 3], 1, false),
                );

                assert_eq!(d1.cmp(&d2), Greater);
            }

            #[test]
            fn bf_sub_bench() {
                let (a1, a2, resa) = (
                    BitFloat::make(vec![1], 0, false),
                    BitFloat::make(vec![1], 1, false),
                    BitFloat::make(vec![usize::MAX], 0, true),
                );

                assert_eq!(a1 - a2, resa);

                let (b1, b2, resb) = (
                    BitFloat::make(vec![1], 0, true),
                    BitFloat::make(vec![1], 1, false),
                    BitFloat::make(vec![1, 1], 1, true),
                );

                assert_eq!(b1 - b2, resb);

                let (c1, c2, resc) = (
                    BitFloat::make(vec![1], 0, false),
                    BitFloat::make(vec![usize::MAX], -1, true),
                    BitFloat::make(vec![usize::MAX, 1], 0, false),
                );

                assert_eq!(c1 - c2, resc);

                let (d1, d2, resd) = (
                    BitFloat::make(vec![usize::MAX], 0, false),
                    BitFloat::make(vec![usize::MAX, usize::MAX], 0, false),
                    BitFloat::make(vec![usize::MAX], -1, true),
                );

                assert_eq!(d1 - d2, resd);

                let (e1, e2, rese) = (
                    BitFloat::make(vec![usize::MAX], 0, true),
                    BitFloat::make(vec![usize::MAX, usize::MAX], 0, true),
                    BitFloat::make(vec![usize::MAX], -1, false),
                );

                assert_eq!(e1 - e2, rese);
            }
        }
    }
}
