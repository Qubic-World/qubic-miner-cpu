#define AVX512 0
#define NUMBER_OF_NEURONS 19000
#define PORT 21841
#define SOLUTION_THRESHOLD 28
#define VERSION_A 1
#define VERSION_B 74
#define VERSION_C 1

#include <intrin.h>
#include <stdio.h>
#include <string.h>
#include <winsock2.h>

#pragma comment (lib, "ws2_32.lib")

#define EQUAL(a, b) (_mm256_movemask_epi8(_mm256_cmpeq_epi64(a, b)) == 0xFFFFFFFF)

#if defined(_MSC_VER)
#define ROL64(a, offset) _rotl64(a, offset)
#else
#define ROL64(a, offset) ((((unsigned long long)a) << offset) ^ (((unsigned long long)a) >> (64 - offset)))
#endif

#if AVX512
const static __m512i zero = _mm512_maskz_set1_epi64(0, 0);
const static __m512i moveThetaPrev = _mm512_setr_epi64(4, 0, 1, 2, 3, 5, 6, 7);
const static __m512i moveThetaNext = _mm512_setr_epi64(1, 2, 3, 4, 0, 5, 6, 7);
const static __m512i rhoB = _mm512_setr_epi64(0, 1, 62, 28, 27, 0, 0, 0);
const static __m512i rhoG = _mm512_setr_epi64(36, 44, 6, 55, 20, 0, 0, 0);
const static __m512i rhoK = _mm512_setr_epi64(3, 10, 43, 25, 39, 0, 0, 0);
const static __m512i rhoM = _mm512_setr_epi64(41, 45, 15, 21, 8, 0, 0, 0);
const static __m512i rhoS = _mm512_setr_epi64(18, 2, 61, 56, 14, 0, 0, 0);
const static __m512i pi1B = _mm512_setr_epi64(0, 3, 1, 4, 2, 5, 6, 7);
const static __m512i pi1G = _mm512_setr_epi64(1, 4, 2, 0, 3, 5, 6, 7);
const static __m512i pi1K = _mm512_setr_epi64(2, 0, 3, 1, 4, 5, 6, 7);
const static __m512i pi1M = _mm512_setr_epi64(3, 1, 4, 2, 0, 5, 6, 7);
const static __m512i pi1S = _mm512_setr_epi64(4, 2, 0, 3, 1, 5, 6, 7);
const static __m512i pi2S1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 8, 10);
const static __m512i pi2S2 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 9, 11);
const static __m512i pi2BG = _mm512_setr_epi64(0, 1, 8, 9, 6, 5, 6, 7);
const static __m512i pi2KM = _mm512_setr_epi64(2, 3, 10, 11, 7, 5, 6, 7);
const static __m512i pi2S3 = _mm512_setr_epi64(4, 5, 12, 13, 4, 5, 6, 7);
const static __m512i padding = _mm512_maskz_set1_epi64(1, 0x8000000000000000);

const static __m512i K12RoundConst0 = _mm512_maskz_set1_epi64(1, 0x000000008000808bULL);
const static __m512i K12RoundConst1 = _mm512_maskz_set1_epi64(1, 0x800000000000008bULL);
const static __m512i K12RoundConst2 = _mm512_maskz_set1_epi64(1, 0x8000000000008089ULL);
const static __m512i K12RoundConst3 = _mm512_maskz_set1_epi64(1, 0x8000000000008003ULL);
const static __m512i K12RoundConst4 = _mm512_maskz_set1_epi64(1, 0x8000000000008002ULL);
const static __m512i K12RoundConst5 = _mm512_maskz_set1_epi64(1, 0x8000000000000080ULL);
const static __m512i K12RoundConst6 = _mm512_maskz_set1_epi64(1, 0x000000000000800aULL);
const static __m512i K12RoundConst7 = _mm512_maskz_set1_epi64(1, 0x800000008000000aULL);
const static __m512i K12RoundConst8 = _mm512_maskz_set1_epi64(1, 0x8000000080008081ULL);
const static __m512i K12RoundConst9 = _mm512_maskz_set1_epi64(1, 0x8000000000008080ULL);
const static __m512i K12RoundConst10 = _mm512_maskz_set1_epi64(1, 0x0000000080000001ULL);
const static __m512i K12RoundConst11 = _mm512_maskz_set1_epi64(1, 0x8000000080008008ULL);

#else

#define KeccakF1600RoundConstant0   0x000000008000808bULL
#define KeccakF1600RoundConstant1   0x800000000000008bULL
#define KeccakF1600RoundConstant2   0x8000000000008089ULL
#define KeccakF1600RoundConstant3   0x8000000000008003ULL
#define KeccakF1600RoundConstant4   0x8000000000008002ULL
#define KeccakF1600RoundConstant5   0x8000000000000080ULL
#define KeccakF1600RoundConstant6   0x000000000000800aULL
#define KeccakF1600RoundConstant7   0x800000008000000aULL
#define KeccakF1600RoundConstant8   0x8000000080008081ULL
#define KeccakF1600RoundConstant9   0x8000000000008080ULL
#define KeccakF1600RoundConstant10  0x0000000080000001ULL

#define declareABCDE \
    unsigned long long Aba, Abe, Abi, Abo, Abu; \
    unsigned long long Aga, Age, Agi, Ago, Agu; \
    unsigned long long Aka, Ake, Aki, Ako, Aku; \
    unsigned long long Ama, Ame, Ami, Amo, Amu; \
    unsigned long long Asa, Ase, Asi, Aso, Asu; \
    unsigned long long Bba, Bbe, Bbi, Bbo, Bbu; \
    unsigned long long Bga, Bge, Bgi, Bgo, Bgu; \
    unsigned long long Bka, Bke, Bki, Bko, Bku; \
    unsigned long long Bma, Bme, Bmi, Bmo, Bmu; \
    unsigned long long Bsa, Bse, Bsi, Bso, Bsu; \
    unsigned long long Ca, Ce, Ci, Co, Cu; \
    unsigned long long Da, De, Di, Do, Du; \
    unsigned long long Eba, Ebe, Ebi, Ebo, Ebu; \
    unsigned long long Ega, Ege, Egi, Ego, Egu; \
    unsigned long long Eka, Eke, Eki, Eko, Eku; \
    unsigned long long Ema, Eme, Emi, Emo, Emu; \
    unsigned long long Esa, Ese, Esi, Eso, Esu; \

#define thetaRhoPiChiIotaPrepareTheta(i, A, E) \
    Da = Cu^ROL64(Ce, 1); \
    De = Ca^ROL64(Ci, 1); \
    Di = Ce^ROL64(Co, 1); \
    Do = Ci^ROL64(Cu, 1); \
    Du = Co^ROL64(Ca, 1); \
    A##ba ^= Da; \
    Bba = A##ba; \
    A##ge ^= De; \
    Bbe = ROL64(A##ge, 44); \
    A##ki ^= Di; \
    Bbi = ROL64(A##ki, 43); \
    A##mo ^= Do; \
    Bbo = ROL64(A##mo, 21); \
    A##su ^= Du; \
    Bbu = ROL64(A##su, 14); \
    E##ba =   Bba ^((~Bbe)&  Bbi ); \
    E##ba ^= KeccakF1600RoundConstant##i; \
    Ca = E##ba; \
    E##be =   Bbe ^((~Bbi)&  Bbo ); \
    Ce = E##be; \
    E##bi =   Bbi ^((~Bbo)&  Bbu ); \
    Ci = E##bi; \
    E##bo =   Bbo ^((~Bbu)&  Bba ); \
    Co = E##bo; \
    E##bu =   Bbu ^((~Bba)&  Bbe ); \
    Cu = E##bu; \
    A##bo ^= Do; \
    Bga = ROL64(A##bo, 28); \
    A##gu ^= Du; \
    Bge = ROL64(A##gu, 20); \
    A##ka ^= Da; \
    Bgi = ROL64(A##ka, 3); \
    A##me ^= De; \
    Bgo = ROL64(A##me, 45); \
    A##si ^= Di; \
    Bgu = ROL64(A##si, 61); \
    E##ga =   Bga ^((~Bge)&  Bgi ); \
    Ca ^= E##ga; \
    E##ge =   Bge ^((~Bgi)&  Bgo ); \
    Ce ^= E##ge; \
    E##gi =   Bgi ^((~Bgo)&  Bgu ); \
    Ci ^= E##gi; \
    E##go =   Bgo ^((~Bgu)&  Bga ); \
    Co ^= E##go; \
    E##gu =   Bgu ^((~Bga)&  Bge ); \
    Cu ^= E##gu; \
    A##be ^= De; \
    Bka = ROL64(A##be, 1); \
    A##gi ^= Di; \
    Bke = ROL64(A##gi, 6); \
    A##ko ^= Do; \
    Bki = ROL64(A##ko, 25); \
    A##mu ^= Du; \
    Bko = ROL64(A##mu, 8); \
    A##sa ^= Da; \
    Bku = ROL64(A##sa, 18); \
    E##ka =   Bka ^((~Bke)&  Bki ); \
    Ca ^= E##ka; \
    E##ke =   Bke ^((~Bki)&  Bko ); \
    Ce ^= E##ke; \
    E##ki =   Bki ^((~Bko)&  Bku ); \
    Ci ^= E##ki; \
    E##ko =   Bko ^((~Bku)&  Bka ); \
    Co ^= E##ko; \
    E##ku =   Bku ^((~Bka)&  Bke ); \
    Cu ^= E##ku; \
    A##bu ^= Du; \
    Bma = ROL64(A##bu, 27); \
    A##ga ^= Da; \
    Bme = ROL64(A##ga, 36); \
    A##ke ^= De; \
    Bmi = ROL64(A##ke, 10); \
    A##mi ^= Di; \
    Bmo = ROL64(A##mi, 15); \
    A##so ^= Do; \
    Bmu = ROL64(A##so, 56); \
    E##ma =   Bma ^((~Bme)&  Bmi ); \
    Ca ^= E##ma; \
    E##me =   Bme ^((~Bmi)&  Bmo ); \
    Ce ^= E##me; \
    E##mi =   Bmi ^((~Bmo)&  Bmu ); \
    Ci ^= E##mi; \
    E##mo =   Bmo ^((~Bmu)&  Bma ); \
    Co ^= E##mo; \
    E##mu =   Bmu ^((~Bma)&  Bme ); \
    Cu ^= E##mu; \
    A##bi ^= Di; \
    Bsa = ROL64(A##bi, 62); \
    A##go ^= Do; \
    Bse = ROL64(A##go, 55); \
    A##ku ^= Du; \
    Bsi = ROL64(A##ku, 39); \
    A##ma ^= Da; \
    Bso = ROL64(A##ma, 41); \
    A##se ^= De; \
    Bsu = ROL64(A##se, 2); \
    E##sa =   Bsa ^((~Bse)&  Bsi ); \
    Ca ^= E##sa; \
    E##se =   Bse ^((~Bsi)&  Bso ); \
    Ce ^= E##se; \
    E##si =   Bsi ^((~Bso)&  Bsu ); \
    Ci ^= E##si; \
    E##so =   Bso ^((~Bsu)&  Bsa ); \
    Co ^= E##so; \
    E##su =   Bsu ^((~Bsa)&  Bse ); \
    Cu ^= E##su;

#define copyFromState(state) \
    Aba = state[ 0]; \
    Abe = state[ 1]; \
    Abi = state[ 2]; \
    Abo = state[ 3]; \
    Abu = state[ 4]; \
    Aga = state[ 5]; \
    Age = state[ 6]; \
    Agi = state[ 7]; \
    Ago = state[ 8]; \
    Agu = state[ 9]; \
    Aka = state[10]; \
    Ake = state[11]; \
    Aki = state[12]; \
    Ako = state[13]; \
    Aku = state[14]; \
    Ama = state[15]; \
    Ame = state[16]; \
    Ami = state[17]; \
    Amo = state[18]; \
    Amu = state[19]; \
    Asa = state[20]; \
    Ase = state[21]; \
    Asi = state[22]; \
    Aso = state[23]; \
    Asu = state[24];

#define copyToState(state) \
    state[ 0] = Aba; \
    state[ 1] = Abe; \
    state[ 2] = Abi; \
    state[ 3] = Abo; \
    state[ 4] = Abu; \
    state[ 5] = Aga; \
    state[ 6] = Age; \
    state[ 7] = Agi; \
    state[ 8] = Ago; \
    state[ 9] = Agu; \
    state[10] = Aka; \
    state[11] = Ake; \
    state[12] = Aki; \
    state[13] = Ako; \
    state[14] = Aku; \
    state[15] = Ama; \
    state[16] = Ame; \
    state[17] = Ami; \
    state[18] = Amo; \
    state[19] = Amu; \
    state[20] = Asa; \
    state[21] = Ase; \
    state[22] = Asi; \
    state[23] = Aso; \
    state[24] = Asu;

#define rounds12 \
    Ca = Aba^Aga^Aka^Ama^Asa; \
    Ce = Abe^Age^Ake^Ame^Ase; \
    Ci = Abi^Agi^Aki^Ami^Asi; \
    Co = Abo^Ago^Ako^Amo^Aso; \
    Cu = Abu^Agu^Aku^Amu^Asu; \
    thetaRhoPiChiIotaPrepareTheta(0, A, E) \
    thetaRhoPiChiIotaPrepareTheta(1, E, A) \
    thetaRhoPiChiIotaPrepareTheta(2, A, E) \
    thetaRhoPiChiIotaPrepareTheta(3, E, A) \
    thetaRhoPiChiIotaPrepareTheta(4, A, E) \
    thetaRhoPiChiIotaPrepareTheta(5, E, A) \
    thetaRhoPiChiIotaPrepareTheta(6, A, E) \
    thetaRhoPiChiIotaPrepareTheta(7, E, A) \
    thetaRhoPiChiIotaPrepareTheta(8, A, E) \
    thetaRhoPiChiIotaPrepareTheta(9, E, A) \
    thetaRhoPiChiIotaPrepareTheta(10, A, E) \
    Da = Cu^ROL64(Ce, 1); \
    De = Ca^ROL64(Ci, 1); \
    Di = Ce^ROL64(Co, 1); \
    Do = Ci^ROL64(Cu, 1); \
    Du = Co^ROL64(Ca, 1); \
    Eba ^= Da; \
    Bba = Eba; \
    Ege ^= De; \
    Bbe = ROL64(Ege, 44); \
    Eki ^= Di; \
    Bbi = ROL64(Eki, 43); \
    Emo ^= Do; \
    Bbo = ROL64(Emo, 21); \
    Esu ^= Du; \
    Bbu = ROL64(Esu, 14); \
    Aba =   Bba ^((~Bbe)&  Bbi ); \
    Aba ^= 0x8000000080008008ULL; \
    Abe =   Bbe ^((~Bbi)&  Bbo ); \
    Abi =   Bbi ^((~Bbo)&  Bbu ); \
    Abo =   Bbo ^((~Bbu)&  Bba ); \
    Abu =   Bbu ^((~Bba)&  Bbe ); \
    Ebo ^= Do; \
    Bga = ROL64(Ebo, 28); \
    Egu ^= Du; \
    Bge = ROL64(Egu, 20); \
    Eka ^= Da; \
    Bgi = ROL64(Eka, 3); \
    Eme ^= De; \
    Bgo = ROL64(Eme, 45); \
    Esi ^= Di; \
    Bgu = ROL64(Esi, 61); \
    Aga =   Bga ^((~Bge)&  Bgi ); \
    Age =   Bge ^((~Bgi)&  Bgo ); \
    Agi =   Bgi ^((~Bgo)&  Bgu ); \
    Ago =   Bgo ^((~Bgu)&  Bga ); \
    Agu =   Bgu ^((~Bga)&  Bge ); \
    Ebe ^= De; \
    Bka = ROL64(Ebe, 1); \
    Egi ^= Di; \
    Bke = ROL64(Egi, 6); \
    Eko ^= Do; \
    Bki = ROL64(Eko, 25); \
    Emu ^= Du; \
    Bko = ROL64(Emu, 8); \
    Esa ^= Da; \
    Bku = ROL64(Esa, 18); \
    Aka =   Bka ^((~Bke)&  Bki ); \
    Ake =   Bke ^((~Bki)&  Bko ); \
    Aki =   Bki ^((~Bko)&  Bku ); \
    Ako =   Bko ^((~Bku)&  Bka ); \
    Aku =   Bku ^((~Bka)&  Bke ); \
    Ebu ^= Du; \
    Bma = ROL64(Ebu, 27); \
    Ega ^= Da; \
    Bme = ROL64(Ega, 36); \
    Eke ^= De; \
    Bmi = ROL64(Eke, 10); \
    Emi ^= Di; \
    Bmo = ROL64(Emi, 15); \
    Eso ^= Do; \
    Bmu = ROL64(Eso, 56); \
    Ama =   Bma ^((~Bme)&  Bmi ); \
    Ame =   Bme ^((~Bmi)&  Bmo ); \
    Ami =   Bmi ^((~Bmo)&  Bmu ); \
    Amo =   Bmo ^((~Bmu)&  Bma ); \
    Amu =   Bmu ^((~Bma)&  Bme ); \
    Ebi ^= Di; \
    Bsa = ROL64(Ebi, 62); \
    Ego ^= Do; \
    Bse = ROL64(Ego, 55); \
    Eku ^= Du; \
    Bsi = ROL64(Eku, 39); \
    Ema ^= Da; \
    Bso = ROL64(Ema, 41); \
    Ese ^= De; \
    Bsu = ROL64(Ese, 2); \
    Asa =   Bsa ^((~Bse)&  Bsi ); \
    Ase =   Bse ^((~Bsi)&  Bso ); \
    Asi =   Bsi ^((~Bso)&  Bsu ); \
    Aso =   Bso ^((~Bsu)&  Bsa ); \
    Asu =   Bsu ^((~Bsa)&  Bse );
#endif

#define K12_security        128
#define K12_capacity        (2 * K12_security)
#define K12_capacityInBytes (K12_capacity / 8)
#define K12_rateInBytes     ((1600 - K12_capacity) / 8)
#define K12_chunkSize       8192
#define K12_suffixLeaf      0x0B

typedef struct
{
    unsigned char state[200];
    unsigned char byteIOIndex;
} KangarooTwelve_F;

static void KeccakP1600_Permute_12rounds(unsigned char* state)
{
#if AVX512
    __m512i Baeiou = _mm512_maskz_loadu_epi64(0x1F, state);
    __m512i Gaeiou = _mm512_maskz_loadu_epi64(0x1F, state + 40);
    __m512i Kaeiou = _mm512_maskz_loadu_epi64(0x1F, state + 80);
    __m512i Maeiou = _mm512_maskz_loadu_epi64(0x1F, state + 120);
    __m512i Saeiou = _mm512_maskz_loadu_epi64(0x1F, state + 160);
    __m512i b0, b1, b2, b3, b4, b5;

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst0);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst1);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst2);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst3);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst4);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst5);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst6);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst7);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst8);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst9);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst10);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
    b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
    b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
    b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
    b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
    b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
    b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
    b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
    Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst11);
    Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
    Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
    Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
    Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
    b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
    b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
    b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
    b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
    Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
    Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
    Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
    Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
    Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

    _mm512_mask_storeu_epi64(state, 0x1F, Baeiou);
    _mm512_mask_storeu_epi64(state + 40, 0x1F, Gaeiou);
    _mm512_mask_storeu_epi64(state + 80, 0x1F, Kaeiou);
    _mm512_mask_storeu_epi64(state + 120, 0x1F, Maeiou);
    _mm512_mask_storeu_epi64(state + 160, 0x1F, Saeiou);
#else
    declareABCDE
        unsigned long long* stateAsLanes = (unsigned long long*)state;
    copyFromState(stateAsLanes)
        rounds12
        copyToState(stateAsLanes)
#endif
}

static void KangarooTwelve_F_Absorb(KangarooTwelve_F* instance, unsigned char* data, unsigned long long dataByteLen)
{
    unsigned long long i = 0;
    while (i < dataByteLen)
    {
        if (!instance->byteIOIndex && dataByteLen >= i + K12_rateInBytes)
        {
#if AVX512
            __m512i Baeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state);
            __m512i Gaeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 40);
            __m512i Kaeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 80);
            __m512i Maeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 120);
            __m512i Saeiou = _mm512_maskz_loadu_epi64(0x1F, instance->state + 160);
#else
            declareABCDE
                unsigned long long* stateAsLanes = (unsigned long long*)instance->state;
            copyFromState(stateAsLanes)
#endif
                unsigned long long modifiedDataByteLen = dataByteLen - i;
            while (modifiedDataByteLen >= K12_rateInBytes)
            {
#if AVX512
                Baeiou = _mm512_xor_si512(Baeiou, _mm512_maskz_loadu_epi64(0x1F, data));
                Gaeiou = _mm512_xor_si512(Gaeiou, _mm512_maskz_loadu_epi64(0x1F, data + 40));
                Kaeiou = _mm512_xor_si512(Kaeiou, _mm512_maskz_loadu_epi64(0x1F, data + 80));
                Maeiou = _mm512_xor_si512(Maeiou, _mm512_maskz_loadu_epi64(0x1F, data + 120));
                Saeiou = _mm512_xor_si512(Saeiou, _mm512_maskz_loadu_epi64(0x01, data + 160));
                __m512i b0, b1, b2, b3, b4, b5;

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst0);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst1);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst2);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst3);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst4);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst5);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst6);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst7);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst8);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst9);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst10);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);

                b0 = _mm512_ternarylogic_epi64(_mm512_ternarylogic_epi64(Baeiou, Gaeiou, Kaeiou, 0x96), Maeiou, Saeiou, 0x96);
                b1 = _mm512_permutexvar_epi64(moveThetaPrev, b0);
                b0 = _mm512_rol_epi64(_mm512_permutexvar_epi64(moveThetaNext, b0), 1);
                b2 = _mm512_permutexvar_epi64(pi1K, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Kaeiou, b0, b1, 0x96), rhoK));
                b3 = _mm512_permutexvar_epi64(pi1M, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Maeiou, b0, b1, 0x96), rhoM));
                b4 = _mm512_permutexvar_epi64(pi1S, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Saeiou, b0, b1, 0x96), rhoS));
                b5 = _mm512_permutexvar_epi64(pi1G, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Gaeiou, b0, b1, 0x96), rhoG));
                b0 = _mm512_permutexvar_epi64(pi1B, _mm512_rolv_epi64(_mm512_ternarylogic_epi64(Baeiou, b0, b1, 0x96), rhoB));
                Baeiou = _mm512_xor_si512(_mm512_ternarylogic_epi64(b0, b5, b2, 0xD2), K12RoundConst11);
                Gaeiou = _mm512_ternarylogic_epi64(b5, b2, b3, 0xD2);
                Kaeiou = _mm512_ternarylogic_epi64(b2, b3, b4, 0xD2);
                Maeiou = _mm512_ternarylogic_epi64(b3, b4, b0, 0xD2);
                Saeiou = _mm512_ternarylogic_epi64(b4, b0, b5, 0xD2);
                b0 = _mm512_permutex2var_epi64(_mm512_unpacklo_epi64(Baeiou, Gaeiou), pi2S1, Saeiou);
                b2 = _mm512_permutex2var_epi64(_mm512_unpackhi_epi64(Baeiou, Gaeiou), pi2S2, Saeiou);
                b1 = _mm512_unpacklo_epi64(Kaeiou, Maeiou);
                b3 = _mm512_unpackhi_epi64(Kaeiou, Maeiou);
                Baeiou = _mm512_permutex2var_epi64(b0, pi2BG, b1);
                Gaeiou = _mm512_permutex2var_epi64(b2, pi2BG, b3);
                Kaeiou = _mm512_permutex2var_epi64(b0, pi2KM, b1);
                Maeiou = _mm512_permutex2var_epi64(b2, pi2KM, b3);
                Saeiou = _mm512_mask_blend_epi64(0x10, _mm512_permutex2var_epi64(b0, pi2S3, b1), Saeiou);
#else
                Aba ^= ((unsigned long long*)data)[0];
                Abe ^= ((unsigned long long*)data)[1];
                Abi ^= ((unsigned long long*)data)[2];
                Abo ^= ((unsigned long long*)data)[3];
                Abu ^= ((unsigned long long*)data)[4];
                Aga ^= ((unsigned long long*)data)[5];
                Age ^= ((unsigned long long*)data)[6];
                Agi ^= ((unsigned long long*)data)[7];
                Ago ^= ((unsigned long long*)data)[8];
                Agu ^= ((unsigned long long*)data)[9];
                Aka ^= ((unsigned long long*)data)[10];
                Ake ^= ((unsigned long long*)data)[11];
                Aki ^= ((unsigned long long*)data)[12];
                Ako ^= ((unsigned long long*)data)[13];
                Aku ^= ((unsigned long long*)data)[14];
                Ama ^= ((unsigned long long*)data)[15];
                Ame ^= ((unsigned long long*)data)[16];
                Ami ^= ((unsigned long long*)data)[17];
                Amo ^= ((unsigned long long*)data)[18];
                Amu ^= ((unsigned long long*)data)[19];
                Asa ^= ((unsigned long long*)data)[20];
                rounds12
#endif
                    data += K12_rateInBytes;
                modifiedDataByteLen -= K12_rateInBytes;
            }
#if AVX512
            _mm512_mask_storeu_epi64(instance->state, 0x1F, Baeiou);
            _mm512_mask_storeu_epi64(instance->state + 40, 0x1F, Gaeiou);
            _mm512_mask_storeu_epi64(instance->state + 80, 0x1F, Kaeiou);
            _mm512_mask_storeu_epi64(instance->state + 120, 0x1F, Maeiou);
            _mm512_mask_storeu_epi64(instance->state + 160, 0x1F, Saeiou);
#else
            copyToState(stateAsLanes)
#endif
                i = dataByteLen - modifiedDataByteLen;
        }
        else
        {
            unsigned char partialBlock;
            if ((dataByteLen - i) + instance->byteIOIndex > K12_rateInBytes)
            {
                partialBlock = K12_rateInBytes - instance->byteIOIndex;
            }
            else
            {
                partialBlock = (unsigned char)(dataByteLen - i);
            }
            i += partialBlock;

            if (!instance->byteIOIndex)
            {
                unsigned int j = 0;
                for (; (j + 8) <= (unsigned int)(partialBlock >> 3); j += 8)
                {
                    ((unsigned long long*)instance->state)[j + 0] ^= ((unsigned long long*)data)[j + 0];
                    ((unsigned long long*)instance->state)[j + 1] ^= ((unsigned long long*)data)[j + 1];
                    ((unsigned long long*)instance->state)[j + 2] ^= ((unsigned long long*)data)[j + 2];
                    ((unsigned long long*)instance->state)[j + 3] ^= ((unsigned long long*)data)[j + 3];
                    ((unsigned long long*)instance->state)[j + 4] ^= ((unsigned long long*)data)[j + 4];
                    ((unsigned long long*)instance->state)[j + 5] ^= ((unsigned long long*)data)[j + 5];
                    ((unsigned long long*)instance->state)[j + 6] ^= ((unsigned long long*)data)[j + 6];
                    ((unsigned long long*)instance->state)[j + 7] ^= ((unsigned long long*)data)[j + 7];
                }
                for (; (j + 4) <= (unsigned int)(partialBlock >> 3); j += 4)
                {
                    ((unsigned long long*)instance->state)[j + 0] ^= ((unsigned long long*)data)[j + 0];
                    ((unsigned long long*)instance->state)[j + 1] ^= ((unsigned long long*)data)[j + 1];
                    ((unsigned long long*)instance->state)[j + 2] ^= ((unsigned long long*)data)[j + 2];
                    ((unsigned long long*)instance->state)[j + 3] ^= ((unsigned long long*)data)[j + 3];
                }
                for (; (j + 2) <= (unsigned int)(partialBlock >> 3); j += 2)
                {
                    ((unsigned long long*)instance->state)[j + 0] ^= ((unsigned long long*)data)[j + 0];
                    ((unsigned long long*)instance->state)[j + 1] ^= ((unsigned long long*)data)[j + 1];
                }
                if (j < (unsigned int)(partialBlock >> 3))
                {
                    ((unsigned long long*)instance->state)[j + 0] ^= ((unsigned long long*)data)[j + 0];
                }
                if (partialBlock & 7)
                {
                    unsigned long long lane = 0;
                    memcpy(&lane, data + (partialBlock & 0xFFFFFFF8), partialBlock & 7);
                    ((unsigned long long*)instance->state)[partialBlock >> 3] ^= lane;
                }
            }
            else
            {
                unsigned int _sizeLeft = partialBlock;
                unsigned int _lanePosition = instance->byteIOIndex >> 3;
                unsigned int _offsetInLane = instance->byteIOIndex & 7;
                const unsigned char* _curData = data;
                while (_sizeLeft > 0)
                {
                    unsigned int _bytesInLane = 8 - _offsetInLane;
                    if (_bytesInLane > _sizeLeft)
                    {
                        _bytesInLane = _sizeLeft;
                    }
                    if (_bytesInLane)
                    {
                        unsigned long long lane = 0;
                        memcpy(&lane, (void*)_curData, _bytesInLane);
                        ((unsigned long long*)instance->state)[_lanePosition] ^= (lane << (_offsetInLane << 3));
                    }
                    _sizeLeft -= _bytesInLane;
                    _lanePosition++;
                    _offsetInLane = 0;
                    _curData += _bytesInLane;
                }
            }

            data += partialBlock;
            instance->byteIOIndex += partialBlock;
            if (instance->byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(instance->state);
                instance->byteIOIndex = 0;
            }
        }
    }
}

static void KangarooTwelve(unsigned char* input, unsigned int inputByteLen, unsigned char* output, unsigned int outputByteLen)
{
    KangarooTwelve_F queueNode;
    KangarooTwelve_F finalNode;
    unsigned int blockNumber, queueAbsorbedLen;

    memset(&finalNode, 0, sizeof(KangarooTwelve_F));
    const unsigned int len = inputByteLen ^ ((K12_chunkSize ^ inputByteLen) & -(K12_chunkSize < inputByteLen));
    KangarooTwelve_F_Absorb(&finalNode, input, len);
    input += len;
    inputByteLen -= len;
    if (len == K12_chunkSize && inputByteLen)
    {
        blockNumber = 1;
        queueAbsorbedLen = 0;
        finalNode.state[finalNode.byteIOIndex] ^= 0x03;
        if (++finalNode.byteIOIndex == K12_rateInBytes)
        {
            KeccakP1600_Permute_12rounds(finalNode.state);
            finalNode.byteIOIndex = 0;
        }
        else
        {
            finalNode.byteIOIndex = (finalNode.byteIOIndex + 7) & ~7;
        }

        while (inputByteLen > 0)
        {
            const unsigned int len = K12_chunkSize ^ ((inputByteLen ^ K12_chunkSize) & -(inputByteLen < K12_chunkSize));
            memset(&queueNode, 0, sizeof(KangarooTwelve_F));
            KangarooTwelve_F_Absorb(&queueNode, input, len);
            input += len;
            inputByteLen -= len;
            if (len == K12_chunkSize)
            {
                ++blockNumber;
                queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
                queueNode.state[K12_rateInBytes - 1] ^= 0x80;
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = K12_capacityInBytes;
                KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
            }
            else
            {
                queueAbsorbedLen = len;
            }
        }

        if (queueAbsorbedLen)
        {
            if (++queueNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = 0;
            }
            if (++queueAbsorbedLen == K12_chunkSize)
            {
                ++blockNumber;
                queueAbsorbedLen = 0;
                queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
                queueNode.state[K12_rateInBytes - 1] ^= 0x80;
                KeccakP1600_Permute_12rounds(queueNode.state);
                queueNode.byteIOIndex = K12_capacityInBytes;
                KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
            }
        }
        else
        {
            memset(queueNode.state, 0, sizeof(queueNode.state));
            queueNode.byteIOIndex = 1;
            queueAbsorbedLen = 1;
        }
    }
    else
    {
        if (len == K12_chunkSize)
        {
            blockNumber = 1;
            finalNode.state[finalNode.byteIOIndex] ^= 0x03;
            if (++finalNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(finalNode.state);
                finalNode.byteIOIndex = 0;
            }
            else
            {
                finalNode.byteIOIndex = (finalNode.byteIOIndex + 7) & ~7;
            }

            memset(queueNode.state, 0, sizeof(queueNode.state));
            queueNode.byteIOIndex = 1;
            queueAbsorbedLen = 1;
        }
        else
        {
            blockNumber = 0;
            if (++finalNode.byteIOIndex == K12_rateInBytes)
            {
                KeccakP1600_Permute_12rounds(finalNode.state);
                finalNode.state[0] ^= 0x07;
            }
            else
            {
                finalNode.state[finalNode.byteIOIndex] ^= 0x07;
            }
        }
    }

    if (blockNumber)
    {
        if (queueAbsorbedLen)
        {
            blockNumber++;
            queueNode.state[queueNode.byteIOIndex] ^= K12_suffixLeaf;
            queueNode.state[K12_rateInBytes - 1] ^= 0x80;
            KeccakP1600_Permute_12rounds(queueNode.state);
            KangarooTwelve_F_Absorb(&finalNode, queueNode.state, K12_capacityInBytes);
        }
        unsigned int n = 0;
        for (unsigned long long v = --blockNumber; v && (n < sizeof(unsigned long long)); ++n, v >>= 8)
        {
        }
        unsigned char encbuf[sizeof(unsigned long long) + 1 + 2];
        for (unsigned int i = 1; i <= n; ++i)
        {
            encbuf[i - 1] = (unsigned char)(blockNumber >> (8 * (n - i)));
        }
        encbuf[n] = (unsigned char)n;
        encbuf[++n] = 0xFF;
        encbuf[++n] = 0xFF;
        KangarooTwelve_F_Absorb(&finalNode, encbuf, ++n);
        finalNode.state[finalNode.byteIOIndex] ^= 0x06;
    }
    finalNode.state[K12_rateInBytes - 1] ^= 0x80;
    KeccakP1600_Permute_12rounds(finalNode.state);
    memcpy(output, finalNode.state, outputByteLen);
}

void random(unsigned char* publicKey, unsigned char* nonce, unsigned char* output, unsigned int outputSize)
{
    unsigned char state[200];
    *((__m256i*) & state[0]) = *((__m256i*)publicKey);
    *((__m256i*) & state[32]) = *((__m256i*)nonce);
    memset(&state[64], 0, sizeof(state) - 64);

    for (unsigned int i = 0; i < outputSize / sizeof(state); i++)
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, sizeof(state));
        output += sizeof(state);
    }
    if (outputSize % sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(output, state, outputSize % sizeof(state));
    }
}

typedef struct
{
    unsigned int size;
    unsigned short protocol;
    unsigned short type;
} RequestResponseHeader;

#define REQUEST_MINER_PUBLIC_KEY 21

#define RESPOND_MINER_PUBLIC_KEY 22

typedef struct
{
    unsigned char minerPublicKey[32];
} RespondMinerPublicKey;

#define RESPOND_RESOURCE_TESTING_SOLUTION 23

typedef struct
{
    unsigned char minerPublicKey[32];
    unsigned char nonce[32];
} RespondResourceTestingSolution;

const static __m256i ZERO = _mm256_setzero_si256();

static volatile char state = 0;

static unsigned long long miningData[65536];
static unsigned char minerPublicKey[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static unsigned char nonce[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static volatile long long numberOfMiningIterations = 0;
static volatile long long numberOfFoundSolutions = 0;

BOOL WINAPI ctrlCHandlerRoutine(DWORD dwCtrlType)
{
    state = 1;

    return TRUE;
}

DWORD WINAPI miningThreadProc(LPVOID)
{
    unsigned char nonce[32];
    unsigned short neuronLinks[NUMBER_OF_NEURONS][2];
    unsigned char neuronValues[NUMBER_OF_NEURONS];
    while (!state)
    {
        if (EQUAL(*((__m256i*)minerPublicKey), ZERO))
        {
            Sleep(1000);
        }
        else
        {
            _rdrand64_step((unsigned long long*) & nonce[0]);
            _rdrand64_step((unsigned long long*) & nonce[8]);
            _rdrand64_step((unsigned long long*) & nonce[16]);
            _rdrand64_step((unsigned long long*) & nonce[24]);
            random(minerPublicKey, nonce, (unsigned char*)neuronLinks, sizeof(neuronLinks));
            for (unsigned int i = 0; i < NUMBER_OF_NEURONS; i++)
            {
                neuronLinks[i][0] %= NUMBER_OF_NEURONS;
                neuronLinks[i][1] %= NUMBER_OF_NEURONS;
            }
            memset(neuronValues, 0xFF, sizeof(neuronValues));

            unsigned int limiter = sizeof(miningData) / sizeof(miningData[0]);
            unsigned int score = 0;
            while (true)
            {
                const unsigned int prevValue0 = neuronValues[NUMBER_OF_NEURONS - 1];
                const unsigned int prevValue1 = neuronValues[NUMBER_OF_NEURONS - 2];

                for (unsigned int j = 0; j < NUMBER_OF_NEURONS; j++)
                {
                    neuronValues[j] = ~(neuronValues[neuronLinks[j][0]] & neuronValues[neuronLinks[j][1]]);
                }

                if (neuronValues[NUMBER_OF_NEURONS - 1] != prevValue0
                    && neuronValues[NUMBER_OF_NEURONS - 2] == prevValue1)
                {
                    if (!(miningData[score >> 6] & (1ULL << (score & 63))))
                    {
                        break;
                    }

                    score++;
                }
                else
                {
                    if (neuronValues[NUMBER_OF_NEURONS - 2] != prevValue1
                        && neuronValues[NUMBER_OF_NEURONS - 1] == prevValue0)
                    {
                        if (miningData[score >> 6] & (1ULL << (score & 63)))
                        {
                            break;
                        }

                        score++;
                    }
                    else
                    {
                        if (!(--limiter))
                        {
                            break;
                        }
                    }
                }
            }

            if (score >= SOLUTION_THRESHOLD)
            {
                while (!EQUAL(*((__m256i*)::nonce), ZERO))
                {
                    Sleep(1);
                }
                *((__m256i*)::nonce) = *((__m256i*)nonce);
                _InterlockedIncrement64(&numberOfFoundSolutions);
            }

            _InterlockedIncrement64(&numberOfMiningIterations);
        }
    }

    return 0;
}

bool sendData(SOCKET serverSocket, char* buffer, unsigned int size)
{
    while (size)
    {
        int numberOfBytes;
        if ((numberOfBytes = send(serverSocket, buffer, size, 0)) <= 0)
        {
            return false;
        }
        buffer += numberOfBytes;
        size -= numberOfBytes;
    }

    return true;
}

bool receiveData(SOCKET serverSocket, char* buffer, unsigned int size)
{
    while (size)
    {
        int numberOfBytes;
        if ((numberOfBytes = recv(serverSocket, buffer, size, 0)) <= 0)
        {
            return false;
        }
        buffer += numberOfBytes;
        size -= numberOfBytes;
    }

    return true;
}

int main(int argc, char* argv[])
{
    printf("Qiner %d.%d.%d is launched.\n", VERSION_A, VERSION_B, VERSION_C);

    if (argc < 2)
    {
        printf("The IP address is not specified!\n");
    }
    else
    {
        unsigned char randomSeed[32];
        ZeroMemory(randomSeed, 32);
        randomSeed[0] = 128;
        randomSeed[1] = 80;
        randomSeed[2] = 115;
        randomSeed[3] = 3;
        randomSeed[4] = 212;
        randomSeed[5] = 249;
        randomSeed[6] = 70;
        randomSeed[7] = 112;
        random(randomSeed, randomSeed, (unsigned char*)miningData, sizeof(miningData));

        SetConsoleCtrlHandler(ctrlCHandlerRoutine, TRUE);

        unsigned int numberOfThreads;
        if (argc < 3)
        {
            SYSTEM_INFO systemInfo;
            GetSystemInfo(&systemInfo);
            numberOfThreads = systemInfo.dwNumberOfProcessors;
        }
        else
        {
            numberOfThreads = atoi(argv[2]);
        }
        printf("%d threads are used.\n", numberOfThreads);
        for (unsigned int i = numberOfThreads; i-- > 0; )
        {
            CreateThread(NULL, 0, miningThreadProc, NULL, 0, NULL);
        }

        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);

        unsigned long long timestamp = GetTickCount64();
        long long prevNumberOfMiningIterations = 0;
        while (!state)
        {
            if (EQUAL(*((__m256i*)minerPublicKey), ZERO) || !EQUAL(*((__m256i*)nonce), ZERO))
            {
                SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                if (serverSocket == INVALID_SOCKET)
                {
                    printf("Fail to create a socket (%d)!\n", WSAGetLastError());
                }
                else
                {
                    sockaddr_in addr;
                    ZeroMemory(&addr, sizeof(addr));
                    addr.sin_family = AF_INET;
                    addr.sin_port = htons(PORT);
                    sscanf(argv[1], "%d.%d.%d.%d", &addr.sin_addr.S_un.S_un_b.s_b1, &addr.sin_addr.S_un.S_un_b.s_b2, &addr.sin_addr.S_un.S_un_b.s_b3, &addr.sin_addr.S_un.S_un_b.s_b4);
                    if (connect(serverSocket, (const sockaddr*)&addr, sizeof(addr)))
                    {
                        printf("Fail to connect to %d.%d.%d.%d (%d)!\n", addr.sin_addr.S_un.S_un_b.s_b1, addr.sin_addr.S_un.S_un_b.s_b2, addr.sin_addr.S_un.S_un_b.s_b3, addr.sin_addr.S_un.S_un_b.s_b4, WSAGetLastError());
                    }
                    else
                    {
                        if (EQUAL(*((__m256i*)minerPublicKey), ZERO))
                        {
                            struct
                            {
                                RequestResponseHeader header;
                            } packet = { { sizeof(packet), VERSION_B, REQUEST_MINER_PUBLIC_KEY } };
                            if (sendData(serverSocket, (char*)&packet, packet.header.size))
                            {
                                struct
                                {
                                    RequestResponseHeader header;
                                    RespondMinerPublicKey payload;
                                } packet;
                                if (receiveData(serverSocket, (char*)&packet, sizeof(packet)) && packet.header.type == RESPOND_MINER_PUBLIC_KEY)
                                {
                                    *((__m256i*)minerPublicKey) = *((__m256i*)packet.payload.minerPublicKey);
                                }
                            }
                        }
                        else
                        {
                            struct
                            {
                                RequestResponseHeader header;
                                RespondResourceTestingSolution payload;
                            } packet = { { sizeof(packet), VERSION_B, RESPOND_RESOURCE_TESTING_SOLUTION } };
                            *((__m256i*)packet.payload.minerPublicKey) = *((__m256i*)minerPublicKey);
                            *((__m256i*)packet.payload.nonce) = *((__m256i*)nonce);
                            if (sendData(serverSocket, (char*)&packet, packet.header.size))
                            {
                                *((__m256i*)minerPublicKey) = ZERO;
                                *((__m256i*)nonce) = ZERO;
                            }
                        }
                    }

                    closesocket(serverSocket);
                }
            }

            Sleep(1000);

            unsigned long long delta = GetTickCount64() - timestamp;
            if (delta >= 1000)
            {
                unsigned char id[70 + 1];

                for (int i = 0; i < 32; i++)
                {
                    id[i << 1] = (minerPublicKey[i] >> 4) + L'A';
                    id[(i << 1) + 1] = (minerPublicKey[i] & 0xF) + L'A';
                }
                unsigned char idBytesChecksum[3];
                KangarooTwelve(minerPublicKey, 32, idBytesChecksum, sizeof(idBytesChecksum));
                for (int i = 0; i < sizeof(idBytesChecksum); i++)
                {
                    id[64 + (i << 1)] = (idBytesChecksum[i] >> 4) + L'A';
                    id[65 + (i << 1)] = (idBytesChecksum[i] & 0xF) + L'A';
                }
                id[70] = 0;

                printf("%d it/s (found %d solutions) (%s).\n", (numberOfMiningIterations - prevNumberOfMiningIterations) * 1000 / delta, numberOfFoundSolutions, id);
                prevNumberOfMiningIterations = numberOfMiningIterations;
                timestamp = GetTickCount64();
            }
        }

        WSACleanup();
    }

    printf("Qiner %d.%d.%d is shut down.\n", VERSION_A, VERSION_B, VERSION_C);

    return 0;
}