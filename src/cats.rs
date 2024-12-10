use std::collections::BTreeSet;
use std::ops::{Add, Sub};
use strum_macros::{EnumIter, EnumString};

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Colour {
    Black = 0,
    Blue = 1,
    Red = 2,
    Green = 3,
    Yellow = 4,
    Grey = 5,
    Fuchsia = 6,
    Orange = 7,
    Teal = 8,
    Brown = 9,
    ToBlack = 10,
    ToBlue = 11,
    ToRed = 12,
    ToGreen = 13,
    ToYellow = 14,
    ToGrey = 15,
    ToFuchsia = 16,
    ToOrange = 17,
    ToTeal = 18,
    ToBrown = 19,
    FromBlack = 20,
    FromBlue = 21,
    FromRed = 22,
    FromGreen = 23,
    FromYellow = 24,
    FromGrey = 25,
    FromFuchsia = 26,
    FromOrange = 27,
    FromTeal = 28,
    FromBrown = 29,
    SameBlack = 30,
    SameBlue = 31,
    SameRed = 32,
    SameGreen = 33,
    SameYellow = 34,
    SameGrey = 35,
    SameFuchsia = 36,
    SameOrange = 37,
    SameTeal = 38,
    SameBrown = 39,
    DiffBlack = 40,
    DiffBlue = 41,
    DiffRed = 42,
    DiffGreen = 43,
    DiffYellow = 44,
    DiffGrey = 45,
    DiffFuchsia = 46,
    DiffOrange = 47,
    DiffTeal = 48,
    DiffBrown = 49,
    OrigBlack = 50,
    OrigBlue = 51,
    OrigRed = 52,
    OrigGreen = 53,
    OrigYellow = 54,
    OrigGrey = 55,
    OrigFuchsia = 56,
    OrigOrange = 57,
    OrigTeal = 58,
    OrigBrown = 59,
    NoColour = 100,
    Mixed = 101,
    Transparent = 102,
    DiffShape = 103,    // Naughty overlading of enum
    Same = 104,         // Naughty overlading of enum
}

impl Colour {
    pub fn new(colour: usize) -> Self {
        Colour::from_usize(colour)
    }

    pub fn from_usize(colour: usize) -> Self {
        match colour {
            0 => Self::Black,
            1 => Self::Blue,
            2 => Self::Red,
            3 => Self::Green,
            4 => Self::Yellow,
            5 => Self::Grey,
            6 => Self::Fuchsia,
            7 => Self::Orange,
            8 => Self::Teal,
            9 => Self::Brown,
            10 => Self::ToBlack,
            11 => Self::ToBlue,
            12 => Self::ToRed,
            13 => Self::ToGreen,
            14 => Self::ToYellow,
            15 => Self::ToGrey,
            16 => Self::ToFuchsia,
            17 => Self::ToOrange,
            18 => Self::ToTeal,
            19 => Self::ToBrown,
            20 => Self::FromBlack,
            21 => Self::FromBlue,
            22 => Self::FromRed,
            23 => Self::FromGreen,
            24 => Self::FromYellow,
            25 => Self::FromGrey,
            26 => Self::FromFuchsia,
            27 => Self::FromOrange,
            28 => Self::FromTeal,
            29 => Self::FromBrown,
            30 => Self::SameBlack,
            31 => Self::SameBlue,
            32 => Self::SameRed,
            33 => Self::SameGreen,
            34 => Self::SameYellow,
            35 => Self::SameGrey,
            36 => Self::SameFuchsia,
            37 => Self::SameOrange,
            38 => Self::SameTeal,
            39 => Self::SameBrown,
            40 => Self::DiffBlack,
            41 => Self::DiffBlue,
            42 => Self::DiffRed,
            43 => Self::DiffGreen,
            44 => Self::DiffYellow,
            45 => Self::DiffGrey,
            46 => Self::DiffFuchsia,
            47 => Self::DiffOrange,
            48 => Self::DiffTeal,
            49 => Self::DiffBrown,
            50 => Self::OrigBlack,
            51 => Self::OrigBlue,
            52 => Self::OrigRed,
            53 => Self::OrigGreen,
            54 => Self::OrigYellow,
            55 => Self::OrigGrey,
            56 => Self::OrigFuchsia,
            57 => Self::OrigOrange,
            58 => Self::OrigTeal,
            59 => Self::OrigBrown,
            100 => Self::NoColour,
            101 => Self::Mixed,
            102 => Self::Transparent,
            103 => Self::DiffShape,    // Naughty overlading of enum
            104 => Self::Same,         // Naughty overlading of enum
            _ => todo!()
        }
    }

    pub fn to_usize(self) -> usize {
        match self {
            Self::Black => 0,
            Self::Blue => 1,
            Self::Red => 2,
            Self::Green => 3,
            Self::Yellow => 4,
            Self::Grey => 5,
            Self::Fuchsia => 6,
            Self::Orange => 7,
            Self::Teal => 8,
            Self::Brown => 9,
            Self::ToBlack => 10,
            Self::ToBlue => 11,
            Self::ToRed => 12,
            Self::ToGreen => 13,
            Self::ToYellow => 14,
            Self::ToGrey => 15,
            Self::ToFuchsia => 16,
            Self::ToOrange => 17,
            Self::ToTeal => 18,
            Self::ToBrown => 19,
            Self::FromBlack => 20,
            Self::FromBlue => 21,
            Self::FromRed => 22,
            Self::FromGreen => 23,
            Self::FromYellow => 24,
            Self::FromGrey => 25,
            Self::FromFuchsia => 26,
            Self::FromOrange => 27,
            Self::FromTeal => 28,
            Self::FromBrown => 29,
            Self::SameBlack => 30,
            Self::SameBlue => 31,
            Self::SameRed => 32,
            Self::SameGreen => 33,
            Self::SameYellow => 34,
            Self::SameGrey => 35,
            Self::SameFuchsia => 36,
            Self::SameOrange => 37,
            Self::SameTeal => 38,
            Self::SameBrown => 39,
            Self::DiffBlack => 40,
            Self::DiffBlue => 41,
            Self::DiffRed => 42,
            Self::DiffGreen => 43,
            Self::DiffYellow => 44,
            Self::DiffGrey => 45,
            Self::DiffFuchsia => 46,
            Self::DiffOrange => 47,
            Self::DiffTeal => 48,
            Self::DiffBrown => 49,
            Self::OrigBlack => 50,
            Self::OrigBlue => 51,
            Self::OrigRed => 52,
            Self::OrigGreen => 53,
            Self::OrigYellow => 54,
            Self::OrigGrey => 55,
            Self::OrigFuchsia => 56,
            Self::OrigOrange => 57,
            Self::OrigTeal => 58,
            Self::OrigBrown => 59,
            Self::NoColour => 100,
            Self::Mixed => 101,
            Self::Transparent => 102,    // Naughty overlading of enum
            Self::DiffShape => 103,    // Naughty overlading of enum
            Self::Same => 104,         // Naughty overlading of enum
            //_ => todo!()
        }
    }

    pub fn to_base(self) -> Self {
        Self::from_usize(Self::to_usize(self) % 10)
    }

    pub fn to_base_sub(self, colour: Self) -> Self {
        Self::from_usize(Self::to_usize(self) - Self::to_usize(colour))
    }

    pub fn is_colour(self) -> bool {
        self == Self::Same || Self::to_usize(self) < 100
    }

    pub fn colours() -> Vec<Self> {
        (1..=9).map(Self::from_usize).collect()
    }

    pub fn is_unit(&self) -> bool {
        *self > Self::Black && *self <= Self::Brown
    }

    pub fn is_to(&self) -> bool {
        *self > Self::ToBlack && *self <= Self::ToBrown
    }

    pub fn is_from(&self) -> bool {
        *self > Self::FromBlack && *self <= Self::FromBrown
    }

    pub fn is_same(&self) -> bool {
        *self > Self::SameBlack && *self <= Self::SameBrown
    }

    pub fn is_diff(&self) -> bool {
        *self > Self::DiffBlack && *self <= Self::DiffBrown
    }

    pub fn is_orig(&self) -> bool {
        *self > Self::OrigBlack && *self <= Self::OrigBrown
    }

    pub fn single_colour_vec(v: &[Self]) -> bool {
        if v.is_empty() {
            return false;
        }

        let c = v[0];

        for col in v.iter() {
            if *col != c {
                return false;
            }
        }

        true
    }

    pub fn all_colours() -> BTreeSet<Self> {
        (0 ..= 9).map(|c| Colour::from_usize(c)).collect()
    }
}

impl Add for Colour {
    type Output = Colour;

    fn add(self, other: Colour) -> Self::Output {
        Self::from_usize(Self::to_usize(self) + Self::to_usize(other))
    }
}

impl Sub for Colour {
    type Output = Colour;

    fn sub(self, other: Colour) -> Self::Output {
        Self::from_usize(Self::to_usize(self) - Self::to_usize(other))
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GridCategory {
    /*
    MirrorXInSkewL,
    MirrorYInSkewL,
    MirrorXOutSkewL,
    MirrorYOutSkewL,
    MirrorXInSkewR,
    MirrorYInSkewR,
    MirrorXOutSkewR,
    MirrorYOutSkewR,
    */
    BGGridInBlack,
    BGGridInColoured,
    BGGridOutBlack,
    BGGridOutColoured,
    BlackPatches,
    BlankIn,
    BlankOut,
    BorderBottomIn,
    BorderBottomOut,
    BorderLeftIn,
    BorderLeftOut,
    BorderRightIn,
    BorderRightOut,
    BorderTopIn,
    BorderTopOut,
    DiagonalOutOrigin,
    DiagonalOutNotOrigin,
    Div9In,
    Div9Out,
    Double,
    EmptyOutput,
    EvenRowsIn,
    EvenRowsOut,
    FullyPopulatedIn,
    FullyPopulatedOut,
    GravityDown,
    GravityLeft,
    GravityRight,
    GravityUp,
    GridLikelyhood,
    HasBGShape,
    HasBGShapeColoured,
    IdenticalColours,
    IdenticalNoColours,
    IdenticalNoPixels,
    InEmpty,
    InLessCountOut,
    InLessCountOutColoured,
    InLessThanOut,
    InLine,
    InOutSameShapes,
    InOutSameShapesColoured,
    InOutSameSize,
    InOutShapeCount,
    InOutShapeCountColoured,
    InOutSquare,
    InOutSquareSameSize,
    InOutSquareSameSizeEven,
    InOutSquareSameSizeOdd,
    InSameCountOut,
    InSameCountOutColoured,
    InSameSize,
    InSquare,
    InvTranspose,
    Is3x3In,
    Is3x3Out,
    IsPanelledXIn,
    IsPanelledXOut,
    IsPanelledYIn,
    IsPanelledYOut,
    MirroredX,
    MirroredY,
    MirrorXIn,
    MirrorXOut,
    MirrorYIn,
    MirrorYOut,
    NoColoursIn(usize),
    NoColoursOut(usize),
    NoColouredShapesIn(usize),
    NoColouredShapesOut(usize),
    NoShapesIn(usize),
    NoShapesOut(usize),
    OutLessCountIn,
    OutLessCountInColoured,
    OutLessThanIn,
    OutLine,
    OutSameSize,
    OutSquare,
    OverlayInSame,
    OverlayOutSame,
    OverlayInDiff,
    OverlayOutDiff,
    Rot180,
    Rot270,
    Rot90,
    SameColour,
    ShapeMaxCntIn(usize),
    ShapeMaxCntOut(usize),
    ShapeMinCntIn(usize),
    ShapeMinCntOut(usize),
    SingleColouredShapeIn,
    SingleColouredShapeOut,
    SingleColourCountIn(usize),
    SingleColourCountOut(usize),
    SingleColourIn2xOut,
    SingleColourIn4xOut,
    SingleColourOut2xIn,
    SingleColourOut4xIn,
    SingleColourIn,
    SingleColourOut,
    SinglePixelOut,
    SingleShapeIn,
    SingleShapeOut,
    SquareShapeSide(usize),
    SquareShapeSize(usize),
    SurroundOut,
    SymmetricIn,
    SymmetricInLR,
    SymmetricInUD,
    SymmetricOut,
    SymmetricOutLR,
    SymmetricOutUD,
    Transpose,
//    GridToSize,
//    GridCalculated,
//    Shapes,
}

impl GridCategory {
    pub fn equal(&self, other: &Self) -> bool {
        self.same(other) && self.get_para() == other.get_para()
    }

    pub fn lt(&self, other: &Self) -> bool {
        self.same(other) && self.get_para() < other.get_para()
    }

    pub fn gt(&self, other: &Self) -> bool {
        self.same(other) && self.get_para() > other.get_para()
    }

    pub fn lte(&self, other: &Self) -> bool {
        self.same(other) && self.get_para() <= other.get_para()
    }

    pub fn gte(&self, other: &Self) -> bool {
        self.same(other) && self.get_para() >= other.get_para()
    }

    fn get_para(&self) -> usize {
        match self {
            GridCategory::NoColouredShapesIn(i) => *i,
            GridCategory::NoColouredShapesOut(i) => *i,
            GridCategory::NoColoursIn(i) => *i,
            GridCategory::NoColoursOut(i) => *i,
            GridCategory::NoShapesIn(i) => *i,
            GridCategory::NoShapesOut(i) => *i,
            GridCategory::ShapeMaxCntIn(i) => *i,
            GridCategory::ShapeMaxCntOut(i) => *i,
            GridCategory::ShapeMinCntIn(i) => *i,
            GridCategory::ShapeMinCntOut(i) => *i,
            GridCategory::SquareShapeSide(i) => *i,
            GridCategory::SquareShapeSize(i) => *i,
            _ => usize::MAX
        }
    }

    pub fn same(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShapeCategory {
    ArmBottom(usize),
    ArmLeft(usize),
    ArmRight(usize),
    ArmTop(usize),
    Full,
    HasBorder,
    HasEmptyHole,
    HasHole,
    //Hollow,   //expensive
    HorizontalLine,
    ManyShapes,
    //MirrorX,  //expensive
    //MirrorY,  //expensive
    OpenBottom,
    OpenBottomHole,
    OpenLeft,
    OpenLeftHole,
    OpenRight,
    OpenRightHole,
    OpenTop,
    OpenTopHole,
    Pixel,
    SingleCell,
    SingleShape,
    Square,
    VerticalLine,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
    FromUpLeft,
    FromUpRight,
    FromDownLeft,
    FromDownRight,
    Other,
}

impl Direction {
    pub fn inverse(&self) -> Self {
        match self {
            Self::Up            => Self::Down,
            Self::Down          => Self::Up,
            Self::Left          => Self::Right,
            Self::Right         => Self::Left,
            Self::UpLeft        => Self::DownRight,
            Self::UpRight       => Self::DownLeft,
            Self::DownLeft      => Self::UpRight,
            Self::DownRight     => Self::UpLeft,
            Self::FromUpLeft    => Self::FromDownRight,
            Self::FromUpRight   => Self::FromDownLeft,
            Self::FromDownLeft  => Self::FromUpRight,
            Self::FromDownRight => Self::FromUpLeft,
            Self::Other         => Self::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumIter, EnumString)]
pub enum Transformation {   // TODO mimimise this list, some are equivalent
    NoTrans,
    MirrorX,
    MirrorY,
    Trans,
    Rotate90,
    Rotate180,
    Rotate270,
    Rotate90MirrorX,
    Rotate180MirrorX,
    Rotate270MirrorX,
    Rotate90MirrorY,
    Rotate180MirrorY,
    Rotate270MirrorY,
    MirrorXRotate90,
    MirrorXRotate180,
    MirrorXRotate270,
    MirrorYRotate90,
    MirrorYRotate180,
    MirrorYRotate270,
}

impl Transformation {
    pub fn inverse(&self) -> Self {
        match self {
            Self::NoTrans           => Self::NoTrans,
            Self::MirrorX           => Self::MirrorX,
            Self::MirrorY           => Self::MirrorY,
            Self::Trans             => Self::Trans,
            Self::Rotate90          => Self::Rotate270,
            Self::Rotate180         => Self::Rotate180,
            Self::Rotate270         => Self::Rotate90,
            Self::Rotate90MirrorX   => Self::MirrorXRotate270,
            Self::Rotate180MirrorX  => Self::MirrorXRotate180,
            Self::Rotate270MirrorX  => Self::MirrorXRotate90,
            Self::Rotate90MirrorY   => Self::MirrorYRotate270,
            Self::Rotate180MirrorY  => Self::MirrorYRotate180,
            Self::Rotate270MirrorY  => Self::MirrorYRotate90,
            Self::MirrorXRotate90   => Self::Rotate270MirrorX,
            Self::MirrorXRotate180  => Self::Rotate180MirrorX,
            Self::MirrorXRotate270  => Self::Rotate90MirrorX,
            Self::MirrorYRotate90   => Self::Rotate270MirrorY,
            Self::MirrorYRotate180  => Self::Rotate180MirrorY,
            Self::MirrorYRotate270  => Self::Rotate90MirrorY,
        }
    }
}

/*
impl FromStr for Transformation {
    type Err = ();

    fn from(input: &str) -> Result<Self, Self::Err> {
        match input {
            "NoTrans"        	=> Ok(Self::NoTrans),
            "MirrorX"        	=> Ok(Self::MirrorX),
            "MirrorY"	        => Ok(Self::MirrorY),
            "Trans"	            => Ok(Self::Trans),
            "Rotate90"	        => Ok(Self::Rotate90),
            "Rotate180"	        => Ok(Self::Rotate180),
            "Rotate270"	        => Ok(Self::Rotate270),
            "Rotate90MirrorX"	=> Ok(Self::Rotate90MirrorX),
            "Rotate180MirrorX"	=> Ok(Self::Rotate180MirrorX),
            "Rotate270MirrorX"	=> Ok(Self::Rotate270MirrorX),
            "Rotate90MirrorY"	=> Ok(Self::Rotate90MirrorY),
            "Rotate180MirrorY"	=> Ok(Self::Rotate180MirrorY),
            "Rotate270MirrorY"	=> Ok(Self::Rotate270MirrorY),
            "MirrorXRotate90"	=> Ok(Self::MirrorXRotate90),
            "MirrorXRotate180"	=> Ok(Self::MirrorXRotate180),
            "MirrorXRotate270"	=> Ok(Self::MirrorXRotate270),
            "MirrorYRotate90"	=> Ok(Self::MirrorYRotate90),
            "MirrorYRotate180"	=> Ok(Self::MirrorYRotate180),
            "MirrorYRotate270"	=> Ok(Self::MirrorYRotate270),
            _ => todo!(),
        }
    }
}
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tag {
    Colour,
    Arm,
    Shape,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Action {
    Together,
    Reach,
}

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShapeEdgeCategory {
    //NotExample,
    Above,
    AboveLeft,
    AboveRight,
    Adjacent,
    Below,
    BelowLeft,
    BelowRight,
    CanContain,
    CommonPixel,
    Gravity,
    HasArm,
    Left,
    MirroredX,
    MirroredY,
    Right,
    Rot180,
    Rot270,
    Rot90,
    Same,
    SameColour,
    SamePixelCount,
    SameShape,
    SameSingle,
    SameSize,
    SingleColour,
    SingleDiffColour,
    SinglePixel,
    Symmetric,
    Transposed,
}

/*
#[repr(usize)]
#[derive(Clone)]
pub enum DiffStates<'a> {
    Start,
    CopyToFirst(&'a dyn Fn(&Example)),
    CopyToLast,
    MoveToFirst,
    MoveToLast,
}
*/

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CellCategory {
    CornerTL,
    CornerTR,
    CornerBL,
    CornerBR,
    InternalCornerTL,
    InternalCornerTR,
    InternalCornerBL,
    InternalCornerBR,
    EdgeT,
    EdgeB,
    EdgeL,
    EdgeR,
    PointT,
    PointB,
    PointL,
    PointR,
    //DiagTL,
    //DiagTR,
    //DiagBL,
    //DiagBR,
    StemTB,
    StemLR,
    Middle,
    BG,
}
