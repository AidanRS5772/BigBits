use crate::bitfrac::BitFrac;
use crate::bitint::BitInt;
use crate::ubitint::*;
use once_cell::sync::Lazy;
use std::f64::consts::{LN_2,LN_10};
use std::ops::*;
use std::fmt;

fn add_bf(a: &mut Vec<usize>, b: &[usize], sh: usize) -> isize {
    let mut sstart = a.len() as isize - (b.len() + sh) as isize;

    if sstart < 0 {
        let pad = vec![0; sstart.unsigned_abs()];
        a.splice(0..0, pad);
        sstart = 0;
    }
    let start = sstart.unsigned_abs();

    let mut c: u8 = 0;
    unsafe {
        for (a_elem, &b_elem) in a[start..].iter_mut().zip(b) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, b_elem, &mut c);
        }

        if c != 0 {
            let a_len = a.len();
            for a_elem in &mut a[a_len - sh..] {
                if c == 0 {
                    break;
                }

                #[cfg(target_arch = "aarch64")]
                add_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                add_carry_x86_64(a_elem, &mut c);
            }
        }
    }

    if c == 1 {
        a.push(1);
    }

    return c as isize;
}

fn sub_bf(a: &mut Vec<usize>, b: &[usize], sh: usize) -> isize {
    let mut sstart = a.len() as isize - (b.len() + sh) as isize;

    if sstart < 0 {
        let pad = vec![0; sstart.unsigned_abs()];
        a.splice(0..0, pad);
        sstart = 0;
    }
    let start = sstart.unsigned_abs();

    unsafe {
        let mut c: u8 = 1;
        for (a_elem, &b_elem) in a[start..].iter_mut().zip(b) {
            #[cfg(target_arch = "aarch64")]
            add_with_carry_aarch64(a_elem, !b_elem, &mut c);

            #[cfg(target_arch = "x86_64")]
            add_with_carry_x86_64(a_elem, !b_elem, &mut c);
        }

        if c == 0 {
            let a_len = a.len();
            for a_elem in &mut a[a_len - sh..] {
                #[cfg(target_arch = "aarch64")]
                sub_carry_aarch64(a_elem, &mut c);

                #[cfg(target_arch = "x86_64")]
                sub_carry_x86_64(a_elem, &mut c);
            }
        }
    }

    let a_len = a.len() as isize;
    if let Some(idx) = a.iter().rposition(|&x| x != 0) {
        a.truncate(idx + 1);
        return idx as isize - a_len + 1;
    } else {
        a.clear();
        return 0;
    }
}

fn shl_bf(bf: &mut BitFloat, sh: u128) {
    let div = (sh / 64) as isize;
    let rem = (sh % 64) as u8;
    bf.exp += div;

    let mv_sz = 64 - rem;

    if rem != 0 {
        let mut carry: usize = 0;
        unsafe {
            for elem in &mut bf.m {
                #[cfg(target_arch = "aarch64")]
                shl_carry_aarch64(elem, &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shl_carry_x86_64(elem, &mut carry, rem, mv_sz);
            }
        }

        if carry > 0 {
            bf.m.push(carry);
            bf.exp += 1;
        }
        if bf.m[0] == 0 {
            bf.m.remove(0);
        }
    }
}

fn shr_bf(bf: &mut BitFloat, sh: u128) {
    let div = (sh / 64) as isize;
    let rem = (sh % 64) as u8;
    bf.exp -= div;

    let mv_sz = 64 - rem;

    if rem != 0 {
        let mut carry: usize = 0;
        unsafe {
            for elem in bf.m.iter_mut().rev() {
                #[cfg(target_arch = "aarch64")]
                shr_carry_aarch64(elem, &mut carry, rem, mv_sz);

                #[cfg(target_arch = "x86_64")]
                shr_carry_x86_64(elem, &mut carry, rem, mv_sz);
            }
        }

        if carry != 0 {
            bf.m.insert(0, carry);
        }
        if bf.m[bf.m.len() - 1] == 0 {
            bf.m.pop();
            bf.exp -= 1;
        }
    }
}

pub fn powi_bf(val: &BitFloat, exp: &UBitInt) -> BitFloat {
    if *exp == 0_usize {
        return (&*ONE).clone();
    } else if *exp == 1_usize {
        return val.clone();
    } else if exp.mod2() {
        return powi_bf(&val.mul_stbl(val), &(exp >> 1_u128)).mul_stbl(val);
    } else {
        return powi_bf(&val.mul_stbl(val), &(exp >> 1_u128));
    }
}

pub fn powi_prim(val: &BitFloat, exp: u128) -> BitFloat {
    if exp == 0 {
        return (&*ONE).clone();
    } else if exp == 1 {
        return val.clone();
    } else if exp & 1 == 1 {
        return powi_prim(&val.mul_stbl(val), exp >> 1).mul_stbl(val);
    } else {
        return powi_prim(&val.mul_stbl(val), exp >> 1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BitFloat {
    pub m: Vec<usize>,
    exp: isize,
    sign: bool,
}

pub trait MES {
    fn get_mes(self) -> (Vec<usize>, isize, bool);
}

impl MES for f64 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let bits = self.to_bits();
        let exp_2 = ((bits >> 52) & 0x7FF) as isize - 1023;

        let exp = if exp_2 < 0 {
            exp_2 / 64 - 1
        } else {
            exp_2 / 64
        };

        let mut m_bin = (((bits & 0xFFFFFFFFFFFFF) | 0x10000000000000) as u128) << 12;

        if exp_2 < 0 {
            m_bin <<= 64 + exp_2 % 64;
        } else {
            m_bin <<= exp_2 % 64
        }

        let mut m: Vec<usize> = vec![];
        let m1: usize = (m_bin >> 64) as usize;
        let m2: usize = m_bin as usize;

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
        }

        return (m, exp, self < 0.0);
    }
}

impl MES for f32 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let bits = self.to_bits();
        let exp_2 = ((bits >> 23) & 0xFF) as isize - 127;

        let exp = if exp_2 < 0 {
            exp_2 / 64 - 1
        } else {
            exp_2 / 64
        };

        let mut m_bin = (((bits & 0x7FFFFF) | 0x800000) as u128) << 41;

        if exp_2 < 0 {
            m_bin <<= 64 + exp_2 % 64;
        } else {
            m_bin <<= exp_2 % 64
        }

        let mut m: Vec<usize> = vec![];
        let m1: usize = (m_bin >> 64) as usize;
        let m2: usize = m_bin as usize;

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
        }

        return (m, exp, self < 0.0);
    }
}

impl MES for i128 {
    fn get_mes(self) -> (Vec<usize>, isize, bool) {
        let uval = self.unsigned_abs();

        let m1 = (uval >> 64) as usize;
        let m2 = uval as usize;

        let mut exp: isize = 0;
        let mut m: Vec<usize> = vec![];

        if m2 != 0 {
            m.push(m2);
        }
        if m1 != 0 {
            m.push(m1);
            exp = 1;
        }

        return (m, exp, self < 0);
    }
}

macro_rules! impl_MES_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl MES for $t{
                fn get_mes(self) -> (Vec<usize>, isize, bool){
                    (vec![self as usize], 0, self < 0)
                }
            }
        )*
    };
}

impl_MES_bf_iprim!(i64, isize, i32, i16, i8);

pub static A: Lazy<BitFloat> =
    Lazy::new(|| BitFloat::make(vec![10807183396718731264, 2], 0, false)); // 256/99
pub static B: Lazy<BitFloat> =
    Lazy::new(|| BitFloat::make(vec![15092790605762363392, 5], 0, false)); // 64/11
pub static C: Lazy<BitFloat> = Lazy::new(|| BitFloat::make(vec![4471937957262917632, 4], 0, false)); // 140/33
pub static ONE: Lazy<BitFloat> = Lazy::new(|| BitFloat::make(vec![1], 0, false));

pub static EXP_COEF: [Lazy<BitFloat>; 76] = [
    Lazy::new(|| BitFloat {
        m: vec![6148914691236517205, 6148914691236517205, 384307168202282325],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![9838263505978427528, 9838263505978427528, 38430716820228232],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![995538569057340880, 15050200720455094493, 3888822535380237],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![7274589020954948387, 11471467302275110934, 121286798000399],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![266423393644341025, 8727613941354504036, 39916464229764],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![11060541645347510280, 13544223589377217351, 4044376621743],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![3516388407270259163, 15531466539987870417, 409780950189],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![5055238462145928110, 12943953435005058622, 41519490049],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![13042608627131219207, 7544410141931148236, 4206803866],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![17034805701860533299, 7831667986391723850, 131106064],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![5339237544698960726, 15297121712990325651, 43186973],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![1828813497624054339, 5650189324678286380, 4375755],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![12442043312119557056, 13041168029402285806, 443356],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![1467543526307980588, 7864313189091876392, 44921],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![4134883993876859582, 9077387799489076369, 4551],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![12743138371231799826, 2998804364680987871, 461],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![7655539792347563461, 13383795152941863135, 46],
        exp: -1,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![5765049298578475653, 13545199090878892228, 4],
        exp: -1,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            12163845780057498668,
            11397111331445958376,
            8848599380141092649,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![4290128483024439680, 6379929362636240148, 896550562772751820],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![1855107394709514265, 1481514791590507755, 90839564215339265],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![8893847575764233327, 5926824644296351157, 9203972167852324],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            2069172804675480312,
            8491406046251966890,
            742717959310837863,
            932557354257931,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            5682260799286693319,
            6343606320420317523,
            8721795629399380610,
            94487814947780,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            17617822156190539556,
            2804498680026112143,
            5408797789292330772,
            9573617250287,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            17795630816413702400,
            9782197408911654598,
            14971511503223605861,
            970010231537,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            8271899138191590582,
            18031266900052975559,
            10868668186038876557,
            98282584804,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            8986745137967672295,
            6046929031178398680,
            4232505784209065437,
            15300047288309889654,
            9958107823,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            15383377192221179586,
            16250199668074715424,
            16228790481957165002,
            10116580555874922864,
            1008967271,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            17302281654208816156,
            5240189914879784074,
            2236974445519986678,
            3867655765349773670,
            102229758,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            15770785565393152198,
            12168803141156933468,
            1941950733137076405,
            10358040,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3587593155421369901,
            14451285627878311220,
            16301039127497754992,
            1049488,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3063716099392634494,
            10949331850695980984,
            8410129707334121980,
            106335,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            12706719108704517933,
            10039157759797129393,
            631918281159969390,
            10774,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            8425088592198786229,
            14126650440612199468,
            3531070532059850169,
            11767241821547354834,
            1091,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            13329161416123324051,
            6507358564532177459,
            15743117798596107194,
            11179547551538927506,
            110,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            13082927893424365176,
            11204263451982710209,
            7143474682208842659,
            3813594008211114641,
            11,
        ],
        exp: -2,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            13117077870839965963,
            16193606152868821842,
            15191969386736771942,
            2499159167984571624,
            1,
        ],
        exp: -2,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            14048166407929123820,
            7757978931354591347,
            18282080231109965146,
            2079690570583933901,
            1050920159145472365,
            2122263708906328300,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1827867679967556695,
            12202894903819943720,
            9092321297750078844,
            12281041418677944503,
            3790683717720780699,
            177014565398596043,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            683575917936730752,
            18398233113768525052,
            13774875745713633912,
            8385918721436730867,
            15003647225310907540,
            21787121575392927,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            5690018879100946354,
            2329282011376577207,
            16807066346182507874,
            12385660639055727920,
            4700173508409005384,
            2207496946178326,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            13823864647798904560,
            8603784254505627586,
            1930153553893278782,
            927188197245596130,
            257762601255954469,
            223666203473634,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            1585073172041206050,
            4667349096714279991,
            184980185015288649,
            12708928229792443461,
            5756781133421047703,
            10298658289588900727,
            22662124476736,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            2493710282930845342,
            18391988556480521387,
            10369550104810598076,
            8479847999143745429,
            3736754416972981804,
            17387864112752993472,
            2296153275832,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            6364380370535722603,
            11337641048360527269,
            15003147773973760686,
            231892237789399860,
            6865819775003554521,
            11529814432100717612,
            232648967731,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            11298743161005975279,
            14033004943201714584,
            17959651811817135044,
            8981503090906645235,
            16167524070467198505,
            13582231888453355635,
            23572268783,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            4665630522583819106,
            10873424584167051587,
            14417633820742285814,
            8655945644354939949,
            648800793677443303,
            5598422530295225644,
            2388370174,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            4858294176896216206,
            12832984567351515640,
            2550790257274512773,
            11902417203574535567,
            2341460052919302008,
            673053323330885039,
            241992493,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            10628842223857153975,
            12801287601799108381,
            2493148824962604876,
            7739055612534299922,
            14703686208477155480,
            11444130658128314568,
            15255770429515363007,
            24518965,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            6228073094043035074,
            6235363321960545036,
            17539757248308312353,
            11176088946009270613,
            7905578606953530368,
            2736125921331983690,
            11792623314511503068,
            2484290,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            12109586393974465708,
            14582777075122301239,
            11669788076392612909,
            10911679644659040833,
            13415602765821400741,
            13526228959240721933,
            4945265277345776349,
            251711,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            9033655253088107482,
            18283203018069497081,
            3384133067713572467,
            4320574061110124848,
            11993718758193585001,
            5416830196546089045,
            17892854031840715248,
            16918,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            15507204958425871228,
            5855385332653271062,
            7208095252700311565,
            16319375428387026874,
            11368944606067123715,
            4074473724891056888,
            1169734751601916455,
            2584,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            16346223147867823774,
            11538244253272919899,
            9817963878429679011,
            231970449134616552,
            8455555096077998092,
            11396710396756610404,
            2351531996598289115,
            15133034697440168979,
            261,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3859302890429771175,
            1336707921911547021,
            13605821120883018119,
            9809054249891749466,
            12438153173991760339,
            3685741448998573855,
            5933569979362516718,
            9738942427930570383,
            26,
        ],
        exp: -3,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            9507209564489881694,
            3427035603349853742,
            6205681040396221932,
            4764237890193789910,
            12911957244673226796,
            9257557695964633457,
            3441954641665002470,
            12688467568087222594,
            2,
        ],
        exp: -3,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            3297532833833560080,
            2596966934800754730,
            14806604340936712257,
            16396485324074146801,
            13252767633830960549,
            7146378729889874507,
            10029247314155528692,
            13792047033546503616,
            15128781802103523802,
            5023702440397075314,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            6080886263943527370,
            15452398818990947634,
            14797646674133262478,
            12430979085396584048,
            12662016590458873001,
            6789105135670874777,
            12920521015626350311,
            1886722345204633056,
            11058995610962514472,
            509007477527932490,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            4741400362572069961,
            807529275653161804,
            3191147930645102880,
            16194776066709260683,
            12523155091388228921,
            11047571194843689063,
            10932009954344796347,
            12180599125963705698,
            4708372396273968155,
            51573240105930764,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            18446414810753863612,
            17842469624492234925,
            16412386068008273607,
            10952845853617356372,
            16386882152073118987,
            16598648174084875916,
            14427953212294455355,
            533636747541851468,
            8600219690261228304,
            5225461731803390,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            15703119506346277895,
            12112050809928061596,
            16233593951674595887,
            16362726973160729813,
            6715496803079141896,
            2848843456792729437,
            697431104111632216,
            5955504287074705547,
            12710177069317336732,
            529449967744059,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            10982711109852211790,
            11491245484559311735,
            5478588385956508598,
            15555597424175353515,
            1214597507452583792,
            17035319537836005697,
            5447433012634995325,
            602845283005033074,
            6688268455428141270,
            107288994822451,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            18217149719615044464,
            10025026720915754532,
            6640507864656578779,
            2074239394141567787,
            13853390692291965332,
            14910623253401234270,
            8010640013597498853,
            4561361434934811941,
            11769604251746200744,
            6794154967004,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            14210302068369642120,
            11989405114571366501,
            1477658293862346171,
            110317402024535629,
            2424207109037108317,
            11518455946064680745,
            7081847635651650376,
            16084867012222280501,
            7775567566024260605,
            585133049640,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            6826716537610520384,
            11282743804231400008,
            13600579100096001425,
            1091349162778647264,
            11150174804303157343,
            2857311634788898770,
            7776500954565309402,
            5026380955615772226,
            10359732508297825287,
            56670797890,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            17186488699759104379,
            17324255220243911561,
            4694931560228298441,
            7587119195952534507,
            8847987312684692646,
            14073688484371223687,
            2386969278403165508,
            13674357062386527161,
            657525005610884707,
            4146389911371600840,
            5675699024,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            17169714972952977803,
            16073646126249402688,
            18234360766041253432,
            14047736563022084282,
            6332062760559938182,
            13770452027855995699,
            4658851034585669693,
            15930724830238981382,
            15250407305035833781,
            9948658288033433053,
            573390327,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            7066361274529993896,
            3500951049128658835,
            11216252080235831779,
            3474223848846854798,
            5405657779743284522,
            2026301554856841007,
            16488687668648629364,
            1664769691934572153,
            1136664198711247522,
            18036933016292353690,
            58054076,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            8640854023138083722,
            10134336216634239803,
            2758843572256585682,
            13474991067102629082,
            6015792222106984773,
            14970651838481682297,
            5668361138223597436,
            15064080989960855692,
            6897297643245765074,
            210920551997784788,
            5881031,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            12864606671137943696,
            3546456496714813719,
            8329822618379343517,
            18173300108264686811,
            12486523655931615402,
            9131395443721170263,
            14963752119809609087,
            3166462490587235447,
            2365831097823013527,
            1228952721302703776,
            13795918408746885194,
            595845,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            2359292169998725358,
            16927611160146150218,
            996169385011854734,
            1209447592777055904,
            17575362488219282551,
            7270545216260299345,
            15298256083544707804,
            13466082387246571245,
            15289133678331073132,
            15475506831626672253,
            1947117266013975704,
            60371,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            9275152249489855703,
            11214525003472135860,
            3734160852422602662,
            5633236022007406672,
            447277015670097823,
            16950340049495178396,
            10445215770057913848,
            958275349851393242,
            6327519663560279244,
            1518344481088677245,
            15760382123107442166,
            6116,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            4146645492707374507,
            12861515057104761827,
            2129822888563726183,
            10754397823048760488,
            7224856110562200134,
            14529013658227318179,
            7623672380948177255,
            2545548379629911219,
            10198050467889038923,
            6409862743347505480,
            14139094526736680585,
            619,
        ],
        exp: -4,
        sign: false,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            5256989886643264624,
            11398184150941529190,
            8250265628767120055,
            2783858833409593231,
            13585039312922289789,
            17423348163233942668,
            15417123154000183611,
            4140485882796023029,
            16323808610652721830,
            4205121691685383198,
            14673689361411079433,
            62,
        ],
        exp: -4,
        sign: true,
    }),
    Lazy::new(|| BitFloat {
        m: vec![
            10201977102712580443,
            9210145055195135738,
            5401350320817013612,
            4596550477412013280,
            17128025185258192298,
            289676684116203033,
            4482100210500073577,
            12753501689933487752,
            17043863508031458003,
            15901600601225178134,
            1809298820573170193,
            6687134407023340067,
            6,
        ],
        exp: -4,
        sign: false,
    }),
];

pub static LN2: Lazy<BitFloat> = Lazy::new(|| BitFloat {
    m: vec![
        11984347084478230215,
        1422093535401088651,
        8055813101346797068,
        13082072642624292584,
        6244647834210745012,
        12283666239418668390,
        1610539766922653204,
        13108345345640254017,
        14969741187151316281,
        13192173624344653850,
        15136605589321663129,
        11061941546837445356,
        8993738265650709846,
        18160713731577796669,
        14424048707614453623,
        3933629848819272244,
        3298789118927550466,
        14232806333737558403,
        9588417116965113009,
        5574758635378132811,
        5889232434169595918,
        5232504371925259464,
        3495649791917164373,
        2928498243069854083,
        15414830800753698833,
        8842679877257317978,
        18073811014293484146,
        15908081087750059806,
        13667507815210917832,
        10890892140069879466,
        12786308645202655659,
    ],
    exp: -1,
    sign: false,
});

pub static ONE_FOUR: Lazy<BitFloat> = Lazy::new(|| BitFloat { m: vec![4611686018427387904], exp: -1, sign: false });

pub static ONE_PIXPI_NEG: Lazy<BitFloat> = Lazy::new(|| BitFloat { m: vec![8767610512681879040, 10184645067323742245, 1869045943895531592], exp: -1, sign: true});

pub static LN_COEF: [Lazy<BitFloat>; 4] = [
    Lazy::new(|| BitFloat { m: vec![2097498820159016960], exp: -1, sign: false }),
    Lazy::new(|| BitFloat { m: vec![14050430709142962176], exp: -1, sign: true }),
    Lazy::new(|| BitFloat { m: vec![3361620884099334144, 2], exp: 0, sign: false }),
    Lazy::new(|| BitFloat { m: vec![9855433068824928256, 1], exp: 0, sign: true }),
];

pub static HALF: Lazy<BitFloat> = Lazy::new(|| BitFloat{m: vec![9223372036854775808], exp: -1, sign: false});

pub static TEN: Lazy<BitFloat> = Lazy::new(|| BitFloat{m: vec![10], exp: 0, sign: false});

impl BitFloat {
    pub fn get_m(&self) -> Vec<usize> {
        self.m.clone()
    }

    pub fn get_exp(&self) -> isize {
        self.exp
    }

    pub fn get_sign(&self) -> bool {
        self.sign
    }

    pub fn make(m: Vec<usize>, exp: isize, sign: bool) -> BitFloat {
        BitFloat { m, exp, sign }
    }

    pub fn from<T: MES>(val: T) -> BitFloat {
        let (m, exp, sign) = val.get_mes();
        BitFloat { m, exp, sign }
    }

    pub fn from_bi(val: &BitInt) -> BitFloat {
        let mut m = val.val.data.clone();
        let exp = m.len().saturating_sub(1) as isize;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        }

        BitFloat {
            m,
            exp,
            sign: val.sign,
        }
    }

    pub fn from_frac(val: &BitFrac) -> BitFloat {
        if val.n > val.d {
            let (int, rem) = val.n.div_rem(&val.d);
            let int_bf = BitFloat::from_bi(&BitInt::from_ubi(&int));
            let n_bf = BitFloat::from_bi(&BitInt::from_ubi(&rem));
            let d_bf = BitFloat::from_bi(&BitInt::from_ubi(&val.d));
            let frac_bf = n_bf / d_bf;

            if val.get_sign() {
                return -(frac_bf + int_bf);
            } else {
                return frac_bf + int_bf;
            }
        } else {
            let n_bf = BitFloat::from_bi(&BitInt::from_ubi(&val.n));
            let d_bf = BitFloat::from_bi(&BitInt::from_ubi(&val.d));
            let frac_bf = n_bf / d_bf;

            if val.get_sign() {
                return -frac_bf;
            } else {
                return frac_bf;
            }
        }
    }

    pub fn to(&self) -> Result<f64, String> {
        let exp: i32 = if let Ok(exp) = self.exp.try_into() {
            exp
        } else {
            return Err("BitFloat is too large or to small for f64".to_string());
        };

        let first_val = if let Some(&val) = self.m.get(self.m.len().saturating_sub(1)) {
            val
        } else {
            return Ok(0.0);
        };

        let base = 2_f64.powi(64);
        let mut sc = 2_f64.powi(64 * exp);

        let mut out = sc * first_val as f64;

        if out.is_infinite() {
            return Err("BitFloat is to large for f64".to_string());
        }

        for &e in self.m[..(self.m.len() - 1)].iter().rev() {
            sc /= base;
            out += sc * e as f64;
        }

        if self.sign {
            return Ok(-out);
        } else {
            return Ok(out);
        }
    }

    pub fn to_bi(&self) -> BitInt{
        let mut int = self.abs_floor();
        if int.m.is_empty(){
            return BitInt::default();
        }
        let mut zeros = vec![0_usize; int.m.len().saturating_sub(int.exp as usize + 1)];
        zeros.append(&mut int.m);

        BitInt::make(UBitInt::make(zeros), self.sign)
    }

    pub fn abs_cmp(&self, other: &BitFloat) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != Equal {
            return exp_ord;
        }

        let mut m_ord: std::cmp::Ordering;

        for (&a, b) in self.m.iter().rev().zip(other.m.iter().rev()) {
            m_ord = a.cmp(b);
            if m_ord != Equal {
                return m_ord;
            }
        }

        return self.m.len().cmp(&other.m.len());
    }

    pub fn log2_int(&self) -> i128 {
        let mut out = self.exp as i128 * 64;
        out += 64 - self.m[self.m.len() - 1].leading_zeros() as i128 - 1;
        return out;
    }

    pub fn mut_neg(&mut self) {
        self.sign = !self.sign;
    }

    pub fn abs(self) -> BitFloat {
        let mut val = self;
        val.sign = false;
        val
    }

    pub fn mut_abs(&mut self) {
        self.sign = false;
    }

    pub fn recipricol(&self) -> BitFloat {
        let log2_d = self.log2_int() + 1;
        let sign_d = self.sign;

        let d = (self >> log2_d).abs();

        let mut x = &*C + (&d).mul_stbl((&d).mul_stbl(&*A) - &*B);

        let max_iter = ((64 * self.m.len() + 1) as f64 / 6.6293566200796096).log(3.0) as usize;
        for _ in 0..=max_iter {
            let e = &*ONE - (&d).mul_stbl(&x);
            let y = (&x).mul_stbl(&e);
            x += &y + (&y).mul_stbl(&e);
        }

        x >>= log2_d;

        if x.m.len() > 3 {
            let num_elements_to_remove = (x.m.len() - self.m.len()).max(0).min(x.m.len() - 3);
            x.m.drain(0..num_elements_to_remove);
        }

        if sign_d {
            return -x;
        } else {
            return x;
        }
    }

    pub fn exp_pow(&self) -> BitFloat{
        let a = (((self.m.len()) as f64).sqrt() as i128)*16;
        let log2 = self.log2_int().max(-a);

        let val = self >> (log2 + a);
        let x = (&val).mul_stbl(&val);

        let mut sum = (&*ONE_FOUR).clone();
        let mut term = (&*ONE).clone();

        let exp_lim = self.exp - self.m.len() as isize - ((log2 + a)/64) as isize;
        let mut state = true;

        for i in 0..76{
            term.mul_assign_stbl(&x);
            let adder = (&term).mul_stbl(&*EXP_COEF[i]);
            if adder.exp < exp_lim{
                state = false;
                break
            }
        }

        if state{
            let mut coef = (&*EXP_COEF[75]).clone();
            loop{
                coef.mul_assign_stbl(&*ONE_PIXPI_NEG);
                term.mul_assign_stbl(&x);
                let adder = (&term).mul_stbl(&coef);
                sum += &adder;
                if adder.exp < exp_lim{
                    break
                }
            }
        }

        sum.mul_assign_stbl(&val);
        sum <<= 1;

        let mut out = (&*ONE + &sum) / (&*ONE - &sum);
        for _ in 0..(log2 + a){
            out.mul_assign_stbl(out.clone());
        }

        let desired_length = self.m.len();
        if out.m.len() > 3 {
            let num_elements_to_remove = (out.m.len() - desired_length).max(0).min(out.m.len() - 3);
            out.m.drain(0..num_elements_to_remove);
        }

        out
    }

    pub fn ln(&self) -> BitFloat{
        let n = self.log2_int();
        let s = self >> n;

        let mut x = &*LN_COEF[3]+(&s).mul_stbl(&*LN_COEF[2]+(&s).mul_stbl(&*LN_COEF[1]+(&s).mul_stbl(&*LN_COEF[0])));

        let mut val = (&x).clone();
        let mut exp_val = (&*ONE).clone();

        let max_iter = ((8*self.m.len()) as f64).log(3.0) as usize;
        for _ in 0..=max_iter{
            exp_val *= (&val).exp_pow();
            val = ((&s - &exp_val)/(&s+&exp_val)) << 1_i128;
            x += &val;
        }

        let mut out = n*&*LN2 + x;

        let desired_length = self.m.len();
        if out.m.len() > 3 {
            let num_elements_to_remove = (out.m.len() - desired_length).max(0).min(out.m.len() - 3);
            out.m.drain(0..num_elements_to_remove);
        }

        return out;
    }

    pub fn abs_floor(&self) -> BitFloat{
        let mut out = self.clone();
        if out.exp >= 0{
            out.m.drain(..out.m.len().saturating_sub(out.exp as usize+1));
        }else{
            return BitFloat::default()
        }

        if let Some(idx) = out.m.iter().position(|&x| x != 0){
            out.m.drain(..idx);
            out.exp += idx as isize;
        }else{
            return BitFloat::default();
        }
        
        out
    }

    pub fn floor(&self) -> BitFloat{
        let mut out = self.abs_floor();
        if out.sign{
            out -= &*ONE;
        }

        out
    }

    pub fn ceil(&self) -> BitFloat{
        let mut out = self.abs_floor();
        if !out.sign{
            out += &*ONE;
        }

        out
    }

    pub fn abs_fract(&self) -> BitFloat{
        let mut out = self.clone();
        if out.exp >= 0{
            out.m.truncate(out.m.len().saturating_sub(out.exp as usize+1))
        }
        out.exp = -1;
        out.sign = false;

        if let Some(idx) = out.m.iter().rposition(|&x| x != 0){
            out.m.truncate(idx+1);
            out.exp -= (out.m.len() - idx - 1) as isize;
        }else{
            out.m.clear();
            out.exp = 0;
        }

        out
    }

    pub fn fract(&self) -> BitFloat{
        self - self.floor()
    }

    pub fn round(&self) -> BitFloat{
        if self.abs_fract() > *HALF{
            return self.ceil();
        }else{
            return self.floor();
        }
    }
}

impl fmt::Display for BitFloat{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let int = self.abs_floor().to_bi();
        let mut fract = self.abs_fract();
        println!("fract: {:?}", fract);

        let max_iter = (((fract.m.len()*64) as f64)*LN_2/LN_10) as usize;
        println!("max_iter: {max_iter}");
        let mut fract_str:String = String::default();
        for _ in 0..max_iter{
            fract *= &*TEN;
            if fract.exp == 0{
                if let Some(digit) = fract.m.last(){
                    fract_str.push_str(&digit.to_string());
                }else{
                    break
                }
            }else{
                fract_str.push('0')
            }
            fract = fract.abs_fract();
        }

        println!("fract string len: {}", fract_str.len());

        write!(f,"{}.{}", int.to_string(), fract_str)?;

        Ok(())
    }
}

macro_rules! impl_peq_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BitFloat{
                fn eq(&self, other: &$t) -> bool {
                    *self == BitFloat::from(*other)
                }
            }
        )*
    };
}

impl_peq_bf_iprim!(f64, f32, i128, i64, isize, i32, i16, i8);

impl PartialOrd for BitFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.sign ^ other.sign {
            if self.sign {
                return Some(std::cmp::Ordering::Less);
            } else {
                return Some(std::cmp::Ordering::Greater);
            }
        }

        let exp_ord = self.exp.cmp(&other.exp);
        if exp_ord != std::cmp::Ordering::Equal {
            if self.sign {
                return Some(exp_ord.reverse());
            } else {
                return Some(exp_ord);
            }
        }

        let mut m_ord = std::cmp::Ordering::Equal;

        for (&a, b) in self.m.iter().rev().zip(other.m.iter().rev()) {
            m_ord = a.cmp(b);
            if m_ord != std::cmp::Ordering::Equal {
                break;
            }
        }

        if m_ord == std::cmp::Ordering::Equal {
            m_ord = self.m.len().cmp(&other.m.len());
        }

        if self.sign {
            return Some(m_ord.reverse());
        } else {
            return Some(m_ord);
        }
    }
}

macro_rules! impl_pord_bf_iprim {
    ($($t:ty),*) => {
        $(
            impl PartialOrd<$t> for BitFloat{
                fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                    self.partial_cmp(&BitFloat::from(*other))
                }
            }
        )*
    };
}

impl_pord_bf_iprim!(f64, f32, i128, i64, isize, i32, i16, i8);

impl Ord for BitFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Add for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, rhs.m, self.exp, self.sign),
            Less => (rhs.m, self.m, rhs.exp, rhs.sign),
            Equal => (self.m, rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut m, &b, sh)
        } else {
            add_bf(&mut m, &b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Add for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Add<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Add<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: &BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };

        let exp_adj = if self.sign ^ rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl<T: MES> Add<T> for BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

impl<T: MES> Add<T> for &BitFloat {
    type Output = BitFloat;

    fn add(self, rhs: T) -> Self::Output {
        self + BitFloat::from(rhs)
    }
}

macro_rules! impl_add_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Add<BitFloat> for $t {
                type Output = BitFloat;

                fn add(self, rhs: BitFloat) -> Self::Output {
                    rhs + self
                }
            }

            impl Add<&BitFloat> for $t {
                type Output = BitFloat;

                fn add(self, rhs: &BitFloat) -> Self::Output {
                    rhs + self
                }
            }
        )*
    }
}

impl_add_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

impl AddAssign for BitFloat {
    fn add_assign(&mut self, rhs: Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            }
            Less => {
                let mut a = rhs.m;

                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            }
        }

        if let Some(idx) = self.m.iter().position(|&x| x != 0) {
            self.m.drain(..idx);
        } else {
            self.m.clear();
        }
    }
}

impl AddAssign<&BitFloat> for BitFloat {
    fn add_assign(&mut self, rhs: &Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();
        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            }
            Less => {
                let mut a = rhs.m.clone();

                let exp_adj = if self.sign ^ rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            }
            Equal => {
                if self.sign ^ rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            }
        }

        if let Some(idx) = self.m.iter().position(|&x| x != 0) {
            self.m.drain(..idx);
        } else {
            self.m.clear();
        }
    }
}

impl<T: MES> AddAssign<T> for BitFloat {
    fn add_assign(&mut self, rhs: T) {
        *self += BitFloat::from(rhs);
    }
}

impl Neg for BitFloat {
    type Output = BitFloat;

    fn neg(self) -> Self::Output {
        BitFloat::make(self.m, self.exp, !self.sign)
    }
}

impl Neg for &BitFloat {
    type Output = BitFloat;

    fn neg(self) -> Self::Output {
        BitFloat::make(self.m.clone(), self.exp, !self.sign)
    }
}

impl Sub for BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, !rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ !rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Sub for &BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, !rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ !rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Sub<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: &Self) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m, &rhs.m, self.exp, self.sign),
            Less => (rhs.m.clone(), &self.m, rhs.exp, !rhs.sign),
            Equal => (self.m, &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ !rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl Sub<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: BitFloat) -> Self::Output {
        use std::cmp::Ordering::*;

        let (mut m, b, mut exp, sign) = match self.abs_cmp(&rhs) {
            Greater => (self.m.clone(), &rhs.m, self.exp, self.sign),
            Less => (rhs.m, &self.m, rhs.exp, !rhs.sign),
            Equal => (self.m.clone(), &rhs.m, self.exp, self.sign),
        };
        let sh = (self.exp - rhs.exp).unsigned_abs();

        let exp_adj = if self.sign ^ !rhs.sign {
            sub_bf(&mut m, b, sh)
        } else {
            add_bf(&mut m, b, sh)
        };

        exp += exp_adj;

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat { m, exp, sign }
    }
}

impl<T: MES> Sub<T> for BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: T) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

impl<T: MES> Sub<T> for &BitFloat {
    type Output = BitFloat;

    fn sub(self, rhs: T) -> Self::Output {
        self - BitFloat::from(rhs)
    }
}

macro_rules! impl_sub_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Sub<BitFloat> for $t {
                type Output = BitFloat;

                fn sub(self, rhs: BitFloat) -> Self::Output {
                    -(rhs - self)
                }
            }

            impl Sub<&BitFloat> for $t {
                type Output = BitFloat;

                fn sub(self, rhs: &BitFloat) -> Self::Output {
                    -(rhs - self)
                }
            }
        )*
    }
}

impl_sub_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

impl SubAssign for BitFloat {
    fn sub_assign(&mut self, rhs: Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();

        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            }
            Less => {
                let mut a = rhs.m;

                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = !rhs.sign
            }
            Equal => {
                if self.sign ^ !rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            }
        }

        if let Some(idx) = self.m.iter().position(|&x| x != 0) {
            self.m.drain(..idx);
        } else {
            self.m.clear();
        }
    }
}

impl SubAssign<&BitFloat> for BitFloat {
    fn sub_assign(&mut self, rhs: &Self) {
        use std::cmp::Ordering::*;

        let sh = (self.exp - rhs.exp).unsigned_abs();
        match self.abs_cmp(&rhs) {
            Greater => {
                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut self.m, &rhs.m, sh)
                } else {
                    add_bf(&mut self.m, &rhs.m, sh)
                };

                self.exp += exp_adj;
            }
            Less => {
                let mut a = rhs.m.clone();

                let exp_adj = if self.sign ^ !rhs.sign {
                    sub_bf(&mut a, &self.m, sh)
                } else {
                    add_bf(&mut a, &self.m, sh)
                };

                self.m = a;
                self.exp = rhs.exp + exp_adj;
                self.sign = rhs.sign
            }
            Equal => {
                if self.sign ^ !rhs.sign {
                    self.m = vec![];
                    self.exp = 0;
                    self.sign = false;
                } else {
                    let exp_adj = add_bf(&mut self.m, &rhs.m, sh);
                    self.exp += exp_adj;
                }
            }
        }

        if let Some(idx) = self.m.iter().position(|&x| x != 0) {
            self.m.drain(..idx);
        } else {
            self.m.clear();
        }
    }
}

impl<T: MES> SubAssign<T> for BitFloat {
    fn sub_assign(&mut self, rhs: T) {
        *self += BitFloat::from(rhs);
    }
}

impl Mul for BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul for &BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: &Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl Mul<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: BitFloat) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl<T: MES> Mul<T> for BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: T) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

impl<T: MES> Mul<T> for &BitFloat {
    type Output = BitFloat;

    fn mul(self, rhs: T) -> Self::Output {
        self * BitFloat::from(rhs)
    }
}

macro_rules! impl_mul_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Mul<BitFloat> for $t{
                type Output = BitFloat;

                fn mul(self, rhs: BitFloat) -> Self::Output{
                    rhs * self
                }
            }

            impl Mul<&BitFloat> for $t{
                type Output = BitFloat;

                fn mul(self, rhs: &BitFloat) -> Self::Output{
                    rhs * self
                }
            }
        )*
    }
}

impl_mul_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

impl MulAssign for BitFloat{
    fn mul_assign(&mut self, rhs: Self) {
        if self.m.is_empty() || rhs.m.is_empty() {
            self.m.clear();
            self.exp = 0;
            self.sign = false;
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        self.exp += rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            self.exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        self.m = m;
        self.sign ^= rhs.sign;
    }
}

impl MulAssign<&BitFloat> for BitFloat{
    fn mul_assign(&mut self, rhs: &Self) {
        if self.m.is_empty() || rhs.m.is_empty() {
            self.m.clear();
            self.exp = 0;
            self.sign = false;
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        self.exp += rhs.exp;

        if m.len() > self.m.len() + rhs.m.len() - 1 {
            self.exp += 1;
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        self.m = m;
        self.sign ^= rhs.sign;
    }
}

impl<T: MES> MulAssign<T> for BitFloat{
    fn mul_assign(&mut self, rhs: T) {
        *self *= BitFloat::from(rhs);
    }
}

pub trait MulStbl<RHS = Self> {
    type Output;
    fn mul_stbl(self, rhs: RHS) -> Self::Output;
}

impl MulStbl for BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        let m_len = m.len();
        if m_len > (self.m.len() + rhs.m.len()) - 1 {
            exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl for &BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > (self.m.len() + rhs.m.len()) - 1 {
            exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: &Self) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > (self.m.len() + rhs.m.len()) - 1 {
            exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

impl MulStbl<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn mul_stbl(self, rhs: BitFloat) -> Self::Output {
        if self.m.is_empty() || rhs.m.is_empty() {
            return BitFloat {
                m: vec![],
                exp: 0,
                sign: false,
            };
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        let mut exp = self.exp + rhs.exp;

        if m.len() > (self.m.len() + rhs.m.len()) - 1 {
            exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        BitFloat {
            m,
            exp,
            sign: self.sign ^ rhs.sign,
        }
    }
}

pub trait MulAssignStbl<RHS = Self>{
    fn mul_assign_stbl(&mut self, rhs: RHS);
}

impl MulAssignStbl for BitFloat{
    fn mul_assign_stbl(&mut self, rhs: Self) {
        if self.m.is_empty() || rhs.m.is_empty() {
            self.m.clear();
            self.exp = 0;
            self.sign = false;
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        self.exp += rhs.exp;

        if m.len() > (self.m.len() + rhs.m.len()) - 1 {
            self.exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        self.m = m;
        self.sign ^= rhs.sign;
    }
}

impl MulAssignStbl<&BitFloat> for BitFloat{
    fn mul_assign_stbl(&mut self, rhs: &Self) {
        if self.m.is_empty() || rhs.m.is_empty() {
            self.m.clear();
            self.exp = 0;
            self.sign = false;
        }

        let mut m = mul_ubi(&self.m, &rhs.m);
        self.exp += rhs.exp;

        if m.len() > (self.m.len() + rhs.m.len()) - 1 {
            self.exp += 1;
        }

        let desired_length = self.m.len().max(rhs.m.len());
        if m.len() > 3 {
            let num_elements_to_remove = (m.len() - desired_length).max(0).min(m.len() - 3);
            m.drain(0..num_elements_to_remove);
        }

        if let Some(idx) = m.iter().position(|&x| x != 0) {
            m.drain(..idx);
        } else {
            m.clear();
        }

        self.m = m;
        self.sign ^= rhs.sign;
    }
}

impl<T:MES> MulAssignStbl<T> for BitFloat{
    fn mul_assign_stbl(&mut self, rhs: T) {
        (*self).mul_assign_stbl(BitFloat::from(rhs))
    }
}

impl Shl<i128> for BitFloat {
    type Output = BitFloat;

    fn shl(self, rhs: i128) -> Self::Output {
        let mut bf = self;

        if rhs < 0 {
            shr_bf(&mut bf, rhs.unsigned_abs());
        } else {
            shl_bf(&mut bf, rhs.unsigned_abs())
        }

        return bf;
    }
}

impl Shl<i128> for &BitFloat {
    type Output = BitFloat;

    fn shl(self, rhs: i128) -> Self::Output {
        let mut bf = self.clone();

        if rhs < 0 {
            shr_bf(&mut bf, rhs.unsigned_abs());
        } else {
            shl_bf(&mut bf, rhs.unsigned_abs())
        }

        return bf;
    }
}

macro_rules! impl_shl_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for BitFloat{
                type Output = BitFloat;

                fn shl(self, rhs: $t) -> Self::Output {
                    let mut bf = self;

                    if rhs < 0{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }

                    return bf;
                }
            }

            impl Shl<$t> for &BitFloat{
                type Output = BitFloat;

                fn shl(self, rhs: $t) -> Self::Output {
                    let mut bf = self.clone();

                    if rhs < 0{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }

                    return bf;
                }
            }
        )*
    }
}

impl_shl_for_bitfloat!(i64, isize, i32, i16, i8);

impl ShlAssign<i128> for BitFloat {
    fn shl_assign(&mut self, rhs: i128) {
        if rhs < 0 {
            shr_bf(self, rhs.unsigned_abs());
        } else {
            shl_bf(self, rhs.unsigned_abs());
        }
    }
}

macro_rules! impl_shl_assign_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl ShlAssign<$t> for BitFloat{
                fn shl_assign(&mut self, rhs: $t) {
                    if rhs < 0{
                        shr_bf(self, rhs.unsigned_abs() as u128);
                    }else{
                        shl_bf(self, rhs.unsigned_abs() as u128);
                    }
                }
            }
        )*
    }
}

impl_shl_assign_for_bitfloat!(i64, isize, i32, i16, i8);

impl Shr<i128> for BitFloat {
    type Output = BitFloat;

    fn shr(self, rhs: i128) -> Self::Output {
        let mut bf = self;

        if rhs < 0 {
            shl_bf(&mut bf, rhs.unsigned_abs());
        } else {
            shr_bf(&mut bf, rhs.unsigned_abs());
        }

        return bf;
    }
}

impl Shr<i128> for &BitFloat {
    type Output = BitFloat;

    fn shr(self, rhs: i128) -> Self::Output {
        let mut bf = self.clone();

        if rhs < 0 {
            shl_bf(&mut bf, rhs.unsigned_abs());
        } else {
            shr_bf(&mut bf, rhs.unsigned_abs());
        }

        return bf;
    }
}

macro_rules! impl_shr_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for BitFloat{
                type Output = BitFloat;

                fn shr(self, rhs: $t) -> Self::Output {
                    let mut bf = self;

                    if rhs < 0{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }

                    return bf;
                }
            }

            impl Shr<$t> for &BitFloat{
                type Output = BitFloat;

                fn shr(self, rhs: $t) -> Self::Output {
                    let mut bf = self.clone();

                    if rhs < 0{
                        shl_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(&mut bf, rhs.unsigned_abs() as u128);
                    }

                    return bf;
                }
            }
        )*
    }
}

impl_shr_for_bitfloat!(i64, isize, i32, i16, i8);

impl ShrAssign<i128> for BitFloat {
    fn shr_assign(&mut self, rhs: i128) {
        if rhs < 0 {
            shl_bf(self, rhs.unsigned_abs());
        } else {
            shr_bf(self, rhs.unsigned_abs());
        }
    }
}

macro_rules! impl_shr_assign_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl ShrAssign<$t> for BitFloat{
                fn shr_assign(&mut self, rhs: $t) {
                    if rhs < 0{
                        shl_bf(self, rhs.unsigned_abs() as u128);
                    }else{
                        shr_bf(self, rhs.unsigned_abs() as u128);
                    }
                }
            }
        )*
    }
}

impl_shr_assign_for_bitfloat!(i64, isize, i32, i16, i8);

impl Div for BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul_stbl(rhs.recipricol())
    }
}

impl Div for &BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: Self) -> Self::Output {
        self.mul_stbl(rhs.recipricol())
    }
}

impl Div<&BitFloat> for BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: &BitFloat) -> Self::Output {
        self.mul_stbl(rhs.recipricol())
    }
}

impl Div<BitFloat> for &BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: BitFloat) -> Self::Output {
        self.mul_stbl(rhs.recipricol())
    }
}

impl<T: MES> Div<T> for BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: T) -> Self::Output {
        self.mul_stbl(BitFloat::from(rhs).recipricol())
    }
}

impl<T: MES> Div<T> for &BitFloat {
    type Output = BitFloat;

    fn div(self, rhs: T) -> Self::Output {
        self.mul_stbl(BitFloat::from(rhs).recipricol())
    }
}

macro_rules! impl_div_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl Div<BitFloat> for $t{
                type Output = BitFloat;

                fn div(self, rhs: BitFloat) -> Self::Output{
                    BitFloat::from(self).mul_stbl(rhs.recipricol())
                }
            }

            impl Div<&BitFloat> for $t{
                type Output = BitFloat;

                fn div(self, rhs: &BitFloat) -> Self::Output{
                    BitFloat::from(self).mul_stbl(rhs.recipricol())
                }
            }
        )*
    }
}

impl_div_for_bitfloat!(f64, f32, i128, i64, isize, i32, i16, i8);

pub trait PowI<RHS = Self> {
    type Output;
    fn powi(self, exp: RHS) -> Self::Output;
}

impl PowI<BitInt> for BitFloat {
    type Output = BitFloat;

    fn powi(self, exp: BitInt) -> Self::Output {
        if exp.get_sign() {
            return powi_bf(&self, &exp.unsighned_abs()).recipricol();
        } else {
            return powi_bf(&self, &exp.unsighned_abs());
        }
    }
}

impl PowI<BitInt> for &BitFloat {
    type Output = BitFloat;

    fn powi(self, exp: BitInt) -> Self::Output {
        if exp.get_sign() {
            return powi_bf(&self, &exp.unsighned_abs()).recipricol();
        } else {
            return powi_bf(&self, &exp.unsighned_abs());
        }
    }
}

impl PowI<&BitInt> for BitFloat {
    type Output = BitFloat;

    fn powi(self, exp: &BitInt) -> Self::Output {
        if exp.get_sign() {
            return powi_bf(&self, &exp.clone().unsighned_abs()).recipricol();
        } else {
            return powi_bf(&self, &exp.clone().unsighned_abs());
        }
    }
}

impl PowI<&BitInt> for &BitFloat {
    type Output = BitFloat;

    fn powi(self, exp: &BitInt) -> Self::Output {
        if exp.get_sign() {
            return powi_bf(&self, &exp.clone().unsighned_abs()).recipricol();
        } else {
            return powi_bf(&self, &exp.clone().unsighned_abs());
        }
    }
}

impl PowI<i128> for BitFloat{
    type Output = BitFloat;

    fn powi(self, exp: i128) -> Self::Output {
        if exp < 0{
            return powi_prim(&self, exp.unsigned_abs()).recipricol();
        }else{
            return powi_prim(&self, exp.unsigned_abs())
        }
    }
}

impl PowI<i128> for &BitFloat{
    type Output = BitFloat;

    fn powi(self, exp: i128) -> Self::Output {
        if exp < 0{
            return powi_prim(&self, exp.unsigned_abs()).recipricol();
        }else{
            return powi_prim(&self, exp.unsigned_abs())
        }
    }
}

macro_rules! impl_powi_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl PowI<$t> for BitFloat{
                type Output = BitFloat;
            
                fn powi(self, exp: $t) -> Self::Output {
                    if exp < 0{
                        return powi_prim(&self, exp.unsigned_abs() as u128).recipricol();
                    }else{
                        return powi_prim(&self, exp.unsigned_abs() as u128)
                    }
                }
            }
            
            impl PowI<$t> for &BitFloat{
                type Output = BitFloat;
            
                fn powi(self, exp: $t) -> Self::Output {
                    if exp < 0{
                        return powi_prim(&self, exp.unsigned_abs() as u128).recipricol();
                    }else{
                        return powi_prim(&self, exp.unsigned_abs() as u128)
                    }
                }
            }
        )*
    }
}

impl_powi_for_bitfloat!(i64, isize, i32, i16, i8);

pub trait PowF<RHS = Self>{
    type Output;
    fn powf(self, rhs: RHS) -> Self::Output;
}

impl PowF for BitFloat{
    type Output = BitFloat;

    fn powf(self, rhs: Self) -> Self::Output {
        self.ln().mul_stbl(rhs).exp_pow()
    }
}

impl PowF for &BitFloat{
    type Output = BitFloat;

    fn powf(self, rhs: Self) -> Self::Output {
        self.ln().mul_stbl(rhs).exp_pow()
    }
}

impl PowF<&BitFloat> for BitFloat{
    type Output = BitFloat;

    fn powf(self, rhs: &Self) -> Self::Output {
        self.ln().mul_stbl(rhs).exp_pow()
    }
}

impl PowF<BitFloat> for &BitFloat{
    type Output = BitFloat;

    fn powf(self, rhs: BitFloat) -> Self::Output {
        self.ln().mul_stbl(rhs).exp_pow()
    }
}

macro_rules! impl_powf_for_bitfloat {
    ($($t:ty),*) => {
        $(
            impl PowF<$t> for BitFloat{
                type Output = BitFloat;

                fn powf(self, rhs: $t) -> Self::Output{
                    (self.ln() * rhs).exp_pow()
                }
            }

            impl PowF<$t> for &BitFloat{
                type Output = BitFloat;

                fn powf(self, rhs: $t) -> Self::Output{
                    (self.ln() * rhs).exp_pow()
                }
            }
        )*
    }
}

impl_powf_for_bitfloat!(f64, f32);
