#include <array>
#include <bitset>
#include <intrin.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <ws2tcpip.h>
#include <wbemprov.h>
#include <winsock2.h>
#include <amp.h>

#define LIMIT 10000

__declspec(align(32)) const __m256i allOnes = _mm256_set1_epi32(0xFFFFFFFF);

long long launchTime = GetTickCount64() - 1;
volatile long long numberOfIterations, numberOfGPUIterations, numberOfOwnErrors = 0, numberOf0Elimitations = 0;
long long numberOfAllErrors = 0;

FILETIME performanceTimestamps[12];
long long performanceCounters[12];
long long performanceGPUCounters[12];

int numberOfChanges = 1;

char miner[70];

volatile struct Task {

	int numberOfMiners;
	char topMiners[70][10];
	int topMinerScores[10];
	int currentMinerPlace;
	int currentMinerScore;
	int numberOfErrors;
	int links[LIMIT][3];

} task, prevTask, prevPrevTask;

__declspec(align(32)) __m256i diffMask = _mm256_setzero_si256();

__declspec(align(32)) __m256i aWords[19683][27];
__declspec(align(32)) __m256i bWords[19683 / 243 / 3][27][3];
__declspec(align(32)) __m256i cWords[19683][19683 / 243 / 3][3];

class InstructionSet
{
	// forward declarations
	class InstructionSet_Internal;

public:
	// getters
	static std::string Brand(void) { return CPU_Rep.brand_; }

	static bool AVX2(void) { return CPU_Rep.f_7_EBX_[5]; }
	static bool AVX512F(void) { return CPU_Rep.f_7_EBX_[16]; }
	static bool RDSEED(void) { return CPU_Rep.f_7_EBX_[18]; }
	static bool AVX512PF(void) { return CPU_Rep.f_7_EBX_[26]; }
	static bool AVX512ER(void) { return CPU_Rep.f_7_EBX_[27]; }
	static bool AVX512CD(void) { return CPU_Rep.f_7_EBX_[28]; }
	static bool RDTSCP(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27]; }

private:
	static const InstructionSet_Internal CPU_Rep;

	class InstructionSet_Internal
	{
	public:
		InstructionSet_Internal()
			: nIds_{ 0 },
			nExIds_{ 0 },
			isIntel_{ false },
			isAMD_{ false },
			f_1_ECX_{ 0 },
			f_1_EDX_{ 0 },
			f_7_EBX_{ 0 },
			f_7_ECX_{ 0 },
			f_81_ECX_{ 0 },
			f_81_EDX_{ 0 },
			data_{},
			extdata_{}
		{
			//int cpuInfo[4] = {-1};
			std::array<int, 4> cpui;

			// Calling __cpuid with 0x0 as the function_id argument
			// gets the number of the highest valid function ID.
			__cpuid(cpui.data(), 0);
			nIds_ = cpui[0];

			for (int i = 0; i <= nIds_; ++i)
			{
				__cpuidex(cpui.data(), i, 0);
				data_.push_back(cpui);
			}

			// Capture vendor string
			char vendor[0x20];
			memset(vendor, 0, sizeof(vendor));
			*reinterpret_cast<int*>(vendor) = data_[0][1];
			*reinterpret_cast<int*>(vendor + 4) = data_[0][3];
			*reinterpret_cast<int*>(vendor + 8) = data_[0][2];
			vendor_ = vendor;
			if (vendor_ == "GenuineIntel")
			{
				isIntel_ = true;
			}
			else if (vendor_ == "AuthenticAMD")
			{
				isAMD_ = true;
			}

			// load bitset with flags for function 0x00000001
			if (nIds_ >= 1)
			{
				f_1_ECX_ = data_[1][2];
				f_1_EDX_ = data_[1][3];
			}

			// load bitset with flags for function 0x00000007
			if (nIds_ >= 7)
			{
				f_7_EBX_ = data_[7][1];
				f_7_ECX_ = data_[7][2];
			}

			// Calling __cpuid with 0x80000000 as the function_id argument
			// gets the number of the highest valid extended ID.
			__cpuid(cpui.data(), 0x80000000);
			nExIds_ = cpui[0];

			char brand[0x40];
			memset(brand, 0, sizeof(brand));

			for (int i = 0x80000000; i <= nExIds_; ++i)
			{
				__cpuidex(cpui.data(), i, 0);
				extdata_.push_back(cpui);
			}

			// load bitset with flags for function 0x80000001
			if (nExIds_ >= 0x80000001)
			{
				f_81_ECX_ = extdata_[1][2];
				f_81_EDX_ = extdata_[1][3];
			}

			// Interpret CPU brand string if reported
			if (nExIds_ >= 0x80000004)
			{
				memcpy(brand, extdata_[2].data(), sizeof(cpui));
				memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
				memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
				brand_ = brand;
			}
		};

		int nIds_;
		int nExIds_;
		std::string vendor_;
		std::string brand_;
		bool isIntel_;
		bool isAMD_;
		std::bitset<32> f_1_ECX_;
		std::bitset<32> f_1_EDX_;
		std::bitset<32> f_7_EBX_;
		std::bitset<32> f_7_ECX_;
		std::bitset<32> f_81_ECX_;
		std::bitset<32> f_81_EDX_;
		std::vector<std::array<int, 4>> data_;
		std::vector<std::array<int, 4>> extdata_;
	};
};

// Initialize static member data
const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;

char* number(long long number, char* buffer) {

	char tmp[12];
	int ptr = 0;
	do {

		if (ptr == 3 || ptr == 7) {

			tmp[ptr++] = '\'';
		}
		tmp[ptr++] = number % 10 + '0';

	} while (number /= 10);
	for (int i = 0; i < ptr; i++) {

		buffer[i] = tmp[ptr - i - 1];
	}
	buffer[ptr] = 0;

	return buffer;
}

int exchange(char* dataToSend, int dataToSendSize, char* receivedDataBuffer, int receivedDataBufferSize) {

	SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	sockaddr_in addr;
	ZeroMemory(&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	inet_pton(AF_INET, "88.99.67.51", &addr.sin_addr.s_addr);
	addr.sin_port = htons(21843);
	if (connect(sock, (const sockaddr*)&addr, sizeof(addr))) {

		printf("Failed to connect (%i)!\n", WSAGetLastError());
		closesocket(sock);

		return -1;
	}

	while (dataToSendSize > 0) {

		int numberOfBytes;
		if ((numberOfBytes = send(sock, dataToSend, dataToSendSize, 0)) == SOCKET_ERROR) {

			printf("Failed to send (%i)!\n", WSAGetLastError());
			closesocket(sock);

			return -1;
		}
		dataToSend += numberOfBytes;
		dataToSendSize -= numberOfBytes;
	}

	int totalNumberOfReceivedBytes = 0;

	if (receivedDataBuffer != NULL) {

		while (TRUE) {

			int numberOfBytes;
			if ((numberOfBytes = recv(sock, receivedDataBuffer, receivedDataBufferSize, 0)) == SOCKET_ERROR) {

				printf("Failed to receive (%i)!\n", WSAGetLastError());
				closesocket(sock);

				return -1;
			}
			if (numberOfBytes == 0) {

				break;
			}
			receivedDataBuffer += numberOfBytes;
			receivedDataBufferSize -= numberOfBytes;
			totalNumberOfReceivedBytes += numberOfBytes;
		}
	}

	closesocket(sock);

	return totalNumberOfReceivedBytes;
}

DWORD WINAPI miningProc(LPVOID lpParameter) {

	while (task.numberOfErrors == 0x7FFFFFFF) {

		Sleep(1);
	}
	Sleep(1000);

	__declspec(align(32)) __m256i values[LIMIT][3];

	for (int i = 0; i < 18; i++) {

		values[27 + i][0] = bWords[0][i][0];
		values[27 + i][1] = bWords[0][i][1];
		values[27 + i][2] = bWords[0][i][2];
	}

	while (TRUE) {

		Task task;
		CopyMemory(&task, (const void*)&::task, sizeof(task));

		FILETIME start;
		GetSystemTimePreciseAsFileTime(&start);

		for (int change = 0; change < numberOfChanges; change++) {

			unsigned int numberOfSteps;
			_rdrand32_step(&numberOfSteps);
			numberOfSteps %= LIMIT;
			int neuronToChange = LIMIT - 1;
			unsigned int inputToChange;
			while (numberOfSteps-- > 0) {

				_rdrand32_step(&inputToChange);
				if ((neuronToChange = task.links[neuronToChange][inputToChange % 3]) < 54) {

					neuronToChange = LIMIT - 1;
				}
			}
			_rdrand32_step(&inputToChange);
			unsigned int link;
			_rdrand32_step(&link);
			task.links[neuronToChange][inputToChange % 3] = link % neuronToChange;
		}

		int numberOfNeurons = 0;

		char neuronFlags[LIMIT];

		for (int i = 54; i < LIMIT - 1; i++) {

			neuronFlags[i] = 0;
		}
		neuronFlags[LIMIT - 1] = 1;
		for (int i = LIMIT; i-- > 54; ) {

			if (neuronFlags[i]) {

				neuronFlags[task.links[i][0]] = 1;
				neuronFlags[task.links[i][1]] = 1;
				neuronFlags[task.links[i][2]] = 1;

				numberOfNeurons++;
			}
		}

		int links[LIMIT][3];

		int currentNeuronIndex = 54;
		int mapping[LIMIT];
		for (int i = 0; i < 54; i++) {

			mapping[i] = i;
		}
		for (int i = 54; i < LIMIT; i++) {

			if (neuronFlags[i]) {

				links[currentNeuronIndex][0] = mapping[task.links[i][0]];
				links[currentNeuronIndex][1] = mapping[task.links[i][1]];
				links[currentNeuronIndex][2] = mapping[task.links[i][2]];
				mapping[i] = currentNeuronIndex++;
			}
		}

		int numberOfErrors = 0;

		for (int a = 0; numberOfErrors <= task.numberOfErrors && a < 19683; a++) {

			for (int i = 0; i < 27; i++) {

				values[i][2] = values[i][1] = values[i][0] = aWords[a][i];
			}

			for (int b = 0; b < 19683 / 243 / 3; b++) {

				for (int i = 18; i < 27; i++) {

					values[27 + i][0] = bWords[b][i][0];
					values[27 + i][1] = bWords[b][i][1];
					values[27 + i][2] = bWords[b][i][2];
				}

				for (int i = 54; i < currentNeuronIndex; i++) {

					__declspec(align(32)) const __m256i tmp00 = _mm256_and_si256(values[links[i][0]][0], values[links[i][1]][0]);
					__declspec(align(32)) const __m256i tmp10 = _mm256_and_si256(values[links[i][0]][1], values[links[i][1]][1]);
					__declspec(align(32)) const __m256i tmp20 = _mm256_and_si256(values[links[i][0]][2], values[links[i][1]][2]);
					__declspec(align(32)) const __m256i tmp01 = _mm256_and_si256(tmp00, values[links[i][2]][0]);
					__declspec(align(32)) const __m256i tmp11 = _mm256_and_si256(tmp10, values[links[i][2]][1]);
					__declspec(align(32)) const __m256i tmp21 = _mm256_and_si256(tmp20, values[links[i][2]][2]);
					values[i][0] = _mm256_xor_si256(tmp01, allOnes);
					values[i][1] = _mm256_xor_si256(tmp11, allOnes);
					values[i][2] = _mm256_xor_si256(tmp21, allOnes);
				}

				long long differences0[4], differences1[4], differences2[4];
				*((__m256i*)differences0) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][0], diffMask), cWords[a][b][0]);
				*((__m256i*)differences1) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][1], diffMask), cWords[a][b][1]);
				*((__m256i*)differences2) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][2], diffMask), cWords[a][b][2]);
				numberOfErrors += (int)(_mm_popcnt_u64(differences0[0]) + _mm_popcnt_u64(differences0[1]) + _mm_popcnt_u64(differences0[2]) + _mm_popcnt_u64(differences0[3])
					+ _mm_popcnt_u64(differences1[0]) + _mm_popcnt_u64(differences1[1]) + _mm_popcnt_u64(differences1[2]) + _mm_popcnt_u64(differences1[3])
					+ _mm_popcnt_u64(differences2[0]) + _mm_popcnt_u64(differences2[1]) + _mm_popcnt_u64(differences2[2]) + _mm_popcnt_u64(differences2[3]));
			}
		}

		InterlockedIncrement64(&numberOfIterations);

		BOOL improved = numberOfErrors < task.numberOfErrors ? TRUE : FALSE;
		if (numberOfErrors <= task.numberOfErrors) {

			if (numberOfErrors < task.numberOfErrors) {

				InterlockedAdd64(&numberOfOwnErrors, task.numberOfErrors - numberOfErrors);
			}
			else {

				InterlockedIncrement64(&numberOf0Elimitations);
			}

			do {

				FILETIME finish;
				GetSystemTimePreciseAsFileTime(&finish);
				ULARGE_INTEGER s, f;
				memcpy(&s, &start, sizeof(ULARGE_INTEGER));
				memcpy(&f, &finish, sizeof(ULARGE_INTEGER));

				struct Solution {

					char command;
					char currentMiner[70];
					int numberOfErrors;
					int links[LIMIT][3];

				} solution;
				solution.command = 1;
				CopyMemory(solution.currentMiner, miner, 70);
				solution.numberOfErrors = numberOfErrors;
				CopyMemory(solution.links, (const void*)task.links, sizeof(solution.links));
				if (exchange((char*)&solution, sizeof(solution), NULL, 0) < 0) {

					if (numberOfErrors < task.numberOfErrors) {

						printf("Failed to send a solution!\n");
					}
				}
				else {

					if (numberOfErrors < task.numberOfErrors) {

						char buffer[16], buffer2[16];
						printf("AVX2: Managed to find a solution reducing number of errors by %s (%s neurons)\n\n", number(task.numberOfErrors - numberOfErrors, buffer), number(numberOfNeurons, buffer2));
					}
				}

				Sleep(5000);

			} while (improved && numberOfErrors < ::task.numberOfErrors);
		}
	}
}

DWORD WINAPI miningProcAVX512(LPVOID lpParameter) {

	while (task.numberOfErrors == 0x7FFFFFFF) {

		Sleep(1);
	}
	Sleep(1000);

	__declspec(align(32)) __m256i values[LIMIT][3];

	for (int i = 0; i < 18; i++) {

		values[27 + i][0] = bWords[0][i][0];
		values[27 + i][1] = bWords[0][i][1];
		values[27 + i][2] = bWords[0][i][2];
	}

	while (TRUE) {

		Task task;
		CopyMemory(&task, (const void*)&::task, sizeof(task));

		FILETIME start;
		GetSystemTimePreciseAsFileTime(&start);

		for (int change = 0; change < numberOfChanges; change++) {

			unsigned int numberOfSteps;
			_rdrand32_step(&numberOfSteps);
			numberOfSteps %= LIMIT;
			int neuronToChange = LIMIT - 1;
			unsigned int inputToChange;
			while (numberOfSteps-- > 0) {

				_rdrand32_step(&inputToChange);
				if ((neuronToChange = task.links[neuronToChange][inputToChange % 3]) < 54) {

					neuronToChange = LIMIT - 1;
				}
			}
			_rdrand32_step(&inputToChange);
			unsigned int link;
			_rdrand32_step(&link);
			task.links[neuronToChange][inputToChange % 3] = link % neuronToChange;
		}

		int numberOfNeurons = 0;

		char neuronFlags[LIMIT];

		for (int i = 54; i < LIMIT - 1; i++) {

			neuronFlags[i] = 0;
		}
		neuronFlags[LIMIT - 1] = 1;
		for (int i = LIMIT; i-- > 54; ) {

			if (neuronFlags[i]) {

				neuronFlags[task.links[i][0]] = 1;
				neuronFlags[task.links[i][1]] = 1;
				neuronFlags[task.links[i][2]] = 1;

				numberOfNeurons++;
			}
		}

		int links[LIMIT][3];

		int currentNeuronIndex = 54;
		int mapping[LIMIT];
		for (int i = 0; i < 54; i++) {

			mapping[i] = i;
		}
		for (int i = 54; i < LIMIT; i++) {

			if (neuronFlags[i]) {

				links[currentNeuronIndex][0] = mapping[task.links[i][0]];
				links[currentNeuronIndex][1] = mapping[task.links[i][1]];
				links[currentNeuronIndex][2] = mapping[task.links[i][2]];
				mapping[i] = currentNeuronIndex++;
			}
		}

		int numberOfErrors = 0;

		for (int a = 0; numberOfErrors <= task.numberOfErrors && a < 19683; a++) {

			for (int i = 0; i < 27; i++) {

				values[i][2] = values[i][1] = values[i][0] = aWords[a][i];
			}

			for (int b = 0; b < 19683 / 243 / 3; b++) {

				for (int i = 18; i < 27; i++) {

					values[27 + i][0] = bWords[b][i][0];
					values[27 + i][1] = bWords[b][i][1];
					values[27 + i][2] = bWords[b][i][2];
				}

				for (int i = 54; i < currentNeuronIndex; i++) {

					values[i][0] = _mm256_ternarylogic_epi32(values[links[i][0]][0], values[links[i][1]][0], values[links[i][2]][0], 0x7F);
					values[i][1] = _mm256_ternarylogic_epi32(values[links[i][0]][1], values[links[i][1]][1], values[links[i][2]][1], 0x7F);
					values[i][2] = _mm256_ternarylogic_epi32(values[links[i][0]][2], values[links[i][1]][2], values[links[i][2]][2], 0x7F);
				}

				long long accumulator[4];
				_mm256_storeu_epi64(accumulator, _mm256_add_epi64(_mm256_add_epi64(_mm256_popcnt_epi64(_mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][0], diffMask), cWords[a][b][0])), _mm256_popcnt_epi64(_mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][1], diffMask), cWords[a][b][1]))), _mm256_popcnt_epi64(_mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][2], diffMask), cWords[a][b][2]))));
				numberOfErrors += (int)(accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3]);
			}
		}
		//printf(">>>> %d %d\n", numberOfErrors, task.numberOfErrors);

		InterlockedIncrement64(&numberOfIterations);

		BOOL improved = numberOfErrors < task.numberOfErrors ? TRUE : FALSE;
		if (numberOfErrors <= task.numberOfErrors) {

			if (numberOfErrors < task.numberOfErrors) {

				InterlockedAdd64(&numberOfOwnErrors, task.numberOfErrors - numberOfErrors);
			}
			else {

				InterlockedIncrement64(&numberOf0Elimitations);
			}

			do {

				FILETIME finish;
				GetSystemTimePreciseAsFileTime(&finish);
				ULARGE_INTEGER s, f;
				memcpy(&s, &start, sizeof(ULARGE_INTEGER));
				memcpy(&f, &finish, sizeof(ULARGE_INTEGER));

				struct Solution {

					char command;
					char currentMiner[70];
					int numberOfErrors;
					int links[LIMIT][3];

				} solution;
				solution.command = 1;
				CopyMemory(solution.currentMiner, miner, 70);
				solution.numberOfErrors = numberOfErrors;
				CopyMemory(solution.links, (const void*)task.links, sizeof(solution.links));
				if (exchange((char*)&solution, sizeof(solution), NULL, 0) < 0) {

					if (numberOfErrors < task.numberOfErrors) {

						printf("Failed to send a solution!\n");
					}
				}
				else {

					if (numberOfErrors < task.numberOfErrors) {

						char buffer[16], buffer2[16];
						printf("AVX-512: Managed to find a solution reducing number of errors by %s (%s neurons)\n\n", number(task.numberOfErrors - numberOfErrors, buffer), number(numberOfNeurons, buffer2));
					}
				}

				Sleep(5000);

			} while (improved && numberOfErrors < ::task.numberOfErrors);
		}
	}
}

DWORD WINAPI miningProcGPU(LPVOID lpParameter) {

	while (task.numberOfErrors == 0x7FFFFFFF) {

		Sleep(1);
	}
	Sleep(1000);

	while (TRUE) {

		Task task;
		CopyMemory(&task, (const void*)&::task, sizeof(task));

		FILETIME start;
		GetSystemTimePreciseAsFileTime(&start);

		for (int change = 0; change < numberOfChanges; change++) {

			unsigned int numberOfSteps;
			_rdrand32_step(&numberOfSteps);
			numberOfSteps %= LIMIT;
			int neuronToChange = LIMIT - 1;
			unsigned int inputToChange;
			while (numberOfSteps-- > 0) {

				_rdrand32_step(&inputToChange);
				if ((neuronToChange = task.links[neuronToChange][inputToChange % 3]) < 54) {

					neuronToChange = LIMIT - 1;
				}
			}
			_rdrand32_step(&inputToChange);
			unsigned int link;
			_rdrand32_step(&link);
			task.links[neuronToChange][inputToChange % 3] = link % neuronToChange;
		}

		int numberOfNeurons = 0;

		char neuronFlags[LIMIT];

		for (int i = 54; i < LIMIT - 1; i++) {

			neuronFlags[i] = 0;
		}
		neuronFlags[LIMIT - 1] = 1;
		for (int i = LIMIT; i-- > 54; ) {

			if (neuronFlags[i]) {

				neuronFlags[task.links[i][0]] = 1;
				neuronFlags[task.links[i][1]] = 1;
				neuronFlags[task.links[i][2]] = 1;

				numberOfNeurons++;
			}
		}

		unsigned int links[LIMIT][3];

		int currentNeuronIndex = 54;
		int mapping[LIMIT];
		for (int i = 0; i < 54; i++) {

			mapping[i] = i;
		}
		for (int i = 54; i < LIMIT; i++) {

			if (neuronFlags[i]) {

				links[currentNeuronIndex][0] = mapping[task.links[i][0]];
				links[currentNeuronIndex][1] = mapping[task.links[i][1]];
				links[currentNeuronIndex][2] = mapping[task.links[i][2]];
				mapping[i] = currentNeuronIndex++;
			}
		}

		if (currentNeuronIndex > 1000) {

			printf("Too much data for GPU mining!\n\n");

			Sleep(1000);
		}
		else {

			unsigned int numberOfSuberrors[19683];

			concurrency::array_view<const unsigned int, 2> __links(currentNeuronIndex, 3, &links[0][0]);
			concurrency::array_view<unsigned int, 1> __numberOfSuberrors(19683, numberOfSuberrors);
			__numberOfSuberrors.discard_data();
			parallel_for_each(
				__numberOfSuberrors.extent,
				[=](concurrency::index<1> idx) restrict(amp) {

					const unsigned int maxNeuronIndex = __links.extent[0];
					const unsigned int subtaskIndex = idx[0];

					const unsigned int populationCounts[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

					unsigned int values[1000];

					const int a = subtaskIndex - 9841;
					unsigned int absoluteValueA = a < 0 ? -a : a;
					for (unsigned int i = 0; i < 9; i++) {

						unsigned int remainderA = absoluteValueA % 3;
						absoluteValueA = (absoluteValueA + 1) / 3;
						values[i * 3] = 0;
						values[i * 3 + 1] = 0;
						values[i * 3 + 2] = 0;
						values[i * 3 + ((a < 0 && remainderA != 0) ? (remainderA ^ 3) : remainderA)] = 0x7FFFFFF;
					}

					values[27] = 0x02492492;
					values[28] = 0x04924924;
					values[29] = 0x01249249;
					values[30] = 0x00E07038;
					values[31] = 0x070381C0;
					values[32] = 0x001C0E07;
					values[33] = 0x0003FE00;
					values[34] = 0x07FC0000;
					values[35] = 0x000001FF;

					unsigned int numberOfSuberrors = 0;

					for (int b = -9841; b <= 9841; ) {

						for (unsigned int i = 0; i < 18; i++) {

							values[36 + i] = 0;
						}
						unsigned int cWord = 0;

						for (unsigned int flag = 1; flag < 0x8000000; flag <<= 1, b++) {

							if (b < 0) {

								unsigned int absoluteValueB = (((1 - b) / 3 + 1) / 3 + 1) / 3;
								unsigned int remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[36 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[39 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[42 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[45 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[48 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[51 + (remainderB != 0 ? (remainderB ^ 3) : remainderB)] |= flag;

								if ((a / b) % 3 == 0) {

									cWord |= flag;
								}
							}
							else {

								unsigned int absoluteValueB = (((1 + b) / 3 + 1) / 3 + 1) / 3;
								unsigned int remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[36 + remainderB] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[39 + remainderB] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[42 + remainderB] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[45 + remainderB] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[48 + remainderB] |= flag;
								remainderB = absoluteValueB % 3;
								absoluteValueB = (absoluteValueB + 1) / 3;
								values[51 + remainderB] |= flag;

								if ((b == 0 ? 0 : a / b) % 3 == 0) {

									cWord |= flag;
								}
							}
						}

						for (unsigned int i = 54; i < maxNeuronIndex; i++) {

							values[i] = ~(values[__links[i][0]] & values[__links[i][1]] & values[__links[i][2]]);
						}

						const unsigned int difference = (values[maxNeuronIndex - 1] & 0x7FFFFFF) ^ cWord;
						numberOfSuberrors += populationCounts[difference & 0xF];
						numberOfSuberrors += populationCounts[(difference >> 4) & 0xF];
						numberOfSuberrors += populationCounts[(difference >> 8) & 0xF];
						numberOfSuberrors += populationCounts[(difference >> 12) & 0xF];
						numberOfSuberrors += populationCounts[(difference >> 16) & 0xF];
						numberOfSuberrors += populationCounts[(difference >> 20) & 0xF];
						numberOfSuberrors += populationCounts[difference >> 24];
					}

					__numberOfSuberrors[subtaskIndex] = numberOfSuberrors;
				}
			);
			__numberOfSuberrors.synchronize();

			int numberOfErrors = 0;

			for (int i = 0; i < 19683; i++) {

				numberOfErrors += numberOfSuberrors[i];
			}

			//printf(">>> %d %d\n", numberOfErrors, task.numberOfErrors);

			InterlockedIncrement64(&numberOfGPUIterations);

			BOOL improved = numberOfErrors < task.numberOfErrors ? TRUE : FALSE;
			if (numberOfErrors <= task.numberOfErrors) {

				if (numberOfErrors < task.numberOfErrors) {

					InterlockedAdd64(&numberOfOwnErrors, task.numberOfErrors - numberOfErrors);
				}
				else {

					InterlockedIncrement64(&numberOf0Elimitations);
				}

				do {

					FILETIME finish;
					GetSystemTimePreciseAsFileTime(&finish);
					ULARGE_INTEGER s, f;
					memcpy(&s, &start, sizeof(ULARGE_INTEGER));
					memcpy(&f, &finish, sizeof(ULARGE_INTEGER));

					struct Solution {

						char command;
						char currentMiner[70];
						int numberOfErrors;
						int links[LIMIT][3];

					} solution;
					solution.command = 1;
					CopyMemory(solution.currentMiner, miner, 70);
					solution.numberOfErrors = numberOfErrors;
					CopyMemory(solution.links, (const void*)task.links, sizeof(solution.links));
					if (exchange((char*)&solution, sizeof(solution), NULL, 0) < 0) {

						if (numberOfErrors < task.numberOfErrors) {

							printf("Failed to send a solution!\n");
						}
					}
					else {

						if (numberOfErrors < task.numberOfErrors) {

							char buffer[16], buffer2[16];
							printf("GPU: Managed to find a solution reducing number of errors by %s (%s neurons)\n\n", number(task.numberOfErrors - numberOfErrors, buffer), number(numberOfNeurons, buffer2));
						}
					}

					Sleep(5000);

				} while (improved && numberOfErrors < ::task.numberOfErrors);
			}
		}
	}
}

DWORD WINAPI httpProc(LPVOID lpParameter) {

	SOCKET serverSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	sockaddr_in addr;
	ZeroMemory(&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(21840);
	if (bind(serverSock, (const sockaddr*)&addr, sizeof(addr))) {

		printf("Failed to bind (%i)!\n", WSAGetLastError());
		closesocket(serverSock);

		return -1;
	}

	if (listen(serverSock, SOMAXCONN)) {

		printf("Failed to listen (%i)!\n", WSAGetLastError());
		closesocket(serverSock);

		return -1;
	}

	char request[65536];
	char response[134 + 120000];
	lstrcpy(response, "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: 120000\r\nContent-Type: application/octet-stream\r\n\r\n");

	while (TRUE) {

		SOCKET clientSock;
		int addrlen = sizeof(addr);
		if ((clientSock = accept(serverSock, NULL, NULL)) != INVALID_SOCKET) {

			recv(clientSock, request, sizeof(request), 0);

			CopyMemory(&response[134], (const void*)&task.links, sizeof(task.links));

			char* dataToSend = response;
			int dataToSendSize = sizeof(response);
			while (dataToSendSize > 0) {

				int numberOfBytes;
				if ((numberOfBytes = send(clientSock, dataToSend, dataToSendSize, 0)) == SOCKET_ERROR) {

					break;
				}
				dataToSend += numberOfBytes;
				dataToSendSize -= numberOfBytes;
			}

			closesocket(clientSock);
		}
	}
}

int main(int argc, char* argv[]) {

	if (argc < 2) {

		printf("qiner.exe <MyIdentity> <NumberOfMiningThreads>\n");

		return 0;
	}

	concurrency::accelerator acc;

	if (argc >= 4 && atoi(argv[3]) != 0) {

		std::vector<concurrency::accelerator> accs = concurrency::accelerator::get_all();
		concurrency::accelerator acc_chosen = accs[0];

		for (int i = 0; i < accs.size(); i++) {
			if (accs[i].dedicated_memory > acc_chosen.dedicated_memory) {
				acc_chosen = accs[i];
			}
		}
		if (!concurrency::accelerator::set_default(acc_chosen.device_path)) {

			printf("Failed to switch to GPU!\n");
		}

		acc = concurrency::accelerator(concurrency::accelerator::default_accelerator);
	}

	__declspec(align(32)) __m256i flags[243];

	for (int i = 0, j = 0; j < 8; j++) {

		int fragments[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

		for (fragments[j] = 1; fragments[j] != 0; fragments[j] <<= 1) {

			flags[i] = _mm256_set_epi32(fragments[7], fragments[6], fragments[5], fragments[4], fragments[3], fragments[2], fragments[1], fragments[0]);
			diffMask = _mm256_or_si256(diffMask, flags[i]);

			if (++i == 243) {

				break;
			}
		}
	}

	ZeroMemory(aWords, sizeof(aWords));
	ZeroMemory(bWords, sizeof(bWords));
	ZeroMemory(cWords, sizeof(cWords));

	for (int a = -9841; a <= 9841; a++) {

		int absoluteValueA = a < 0 ? -a : a;
		for (int i = 0; i < 9; i++) {

			int remainderA = absoluteValueA % 3;
			absoluteValueA = (absoluteValueA + 1) / 3;
			aWords[a + 9841][i * 3 + ((a < 0 && remainderA != 0) ? (remainderA ^ 3) : remainderA)] = allOnes;
		}
	}

	for (int b = -9841; b <= 9841; ) {

		for (int i = 0; i < 243; i++) {

			int absoluteValueB0 = b < 0 ? -b : b;
			int absoluteValueB1 = (b + 1) < 0 ? -(b + 1) : (b + 1);
			int absoluteValueB2 = (b + 2) < 0 ? -(b + 2) : (b + 2);
			for (int j = 0; j < 9; j++) {

				int remainderB0 = absoluteValueB0 % 3;
				int remainderB1 = absoluteValueB1 % 3;
				int remainderB2 = absoluteValueB2 % 3;
				absoluteValueB0 = (absoluteValueB0 + 1) / 3;
				absoluteValueB1 = (absoluteValueB1 + 1) / 3;
				absoluteValueB2 = (absoluteValueB2 + 1) / 3;
				bWords[(b + 9841) / 243 / 3][j * 3 + ((b < 0 && remainderB0 != 0) ? (remainderB0 ^ 3) : remainderB0)][0] = _mm256_or_si256(bWords[(b + 9841) / 243 / 3][j * 3 + ((b < 0 && remainderB0 != 0) ? (remainderB0 ^ 3) : remainderB0)][0], flags[i]);
				bWords[((b + 1) + 9841) / 243 / 3][j * 3 + (((b + 1) < 0 && remainderB1 != 0) ? (remainderB1 ^ 3) : remainderB1)][1] = _mm256_or_si256(bWords[((b + 1) + 9841) / 243 / 3][j * 3 + (((b + 1) < 0 && remainderB1 != 0) ? (remainderB1 ^ 3) : remainderB1)][1], flags[i]);
				bWords[((b + 2) + 9841) / 243 / 3][j * 3 + (((b + 2) < 0 && remainderB2 != 0) ? (remainderB2 ^ 3) : remainderB2)][2] = _mm256_or_si256(bWords[((b + 2) + 9841) / 243 / 3][j * 3 + (((b + 2) < 0 && remainderB2 != 0) ? (remainderB2 ^ 3) : remainderB2)][2], flags[i]);
			}

			b += 3;
		}
	}

	for (int a = -9841; a <= 9841; a++) {

		for (int b = -9841; b <= 9841; ) {

			for (int i = 0; i < 243; i++) {

				const int c0 = (b == 0 ? 0 : a / b);
				const int c1 = ((b + 1) == 0 ? 0 : a / (b + 1));
				const int c2 = ((b + 2) == 0 ? 0 : a / (b + 2));

				if ((c0 < 0 ? -c0 : c0) % 3 == 0) {

					cWords[a + 9841][(b + 9841) / 243 / 3][0] = _mm256_or_si256(cWords[a + 9841][(b + 9841) / 243 / 3][0], flags[i]);
				}
				if ((c1 < 0 ? -c1 : c1) % 3 == 0) {

					cWords[a + 9841][((b + 1) + 9841) / 243 / 3][1] = _mm256_or_si256(cWords[a + 9841][((b + 1) + 9841) / 243 / 3][1], flags[i]);
				}
				if ((c2 < 0 ? -c2 : c2) % 3 == 0) {

					cWords[a + 9841][((b + 2) + 9841) / 243 / 3][2] = _mm256_or_si256(cWords[a + 9841][((b + 2) + 9841) / 243 / 3][2], flags[i]);
				}

				b += 3;
			}
		}
	}

	CopyMemory(miner, argv[1], 70);

	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);

	task.numberOfErrors = 0x7FFFFFFF;

	SYSTEM_INFO systemInfo;
	GetNativeSystemInfo(&systemInfo);

	int numberOfThreads = (argc >= 3) ? atoi(argv[2]) : systemInfo.dwNumberOfProcessors;

	for (int i = 0; i < numberOfThreads; i++) {

		CreateThread(NULL, 2 * 1024 * 1024, (InstructionSet::AVX512F() && argc <= 4) ? miningProcAVX512 : miningProc, NULL, 0, NULL);
	}
	if (argc >= 4 && atoi(argv[3]) != 0) {

		CreateThread(NULL, 2 * 1024 * 1024, miningProcGPU, NULL, 0, NULL);
	}

	long long latestSolutionTime = GetTickCount64();

	for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]); i++) {

		GetSystemTimePreciseAsFileTime(&performanceTimestamps[i]);
		performanceCounters[i] = 0;
		performanceGPUCounters[i] = 0;
	}
	numberOfIterations = 0;
	numberOfGPUIterations = 0;

	while (TRUE) {

		while (TRUE) {

			int prevNumberOfErrors = task.numberOfErrors;

			char getTask[71];
			getTask[0] = 0;
			CopyMemory(&getTask[1], miner, 70);
			if (exchange(getTask, sizeof(getTask), (char*)&task, sizeof(task)) != sizeof(task)) {

				printf("Failed to receive a task!\n");

				Sleep(5000);
			}
			else {

				BOOL justChanged = FALSE;

				if (prevNumberOfErrors > task.numberOfErrors) {

					latestSolutionTime = GetTickCount64();

					justChanged = TRUE;

					if (prevNumberOfErrors == 0x7FFFFFFF) {

						CopyMemory((void*)&prevTask, (void*)&task, sizeof(Task));
						CopyMemory((void*)&prevPrevTask, (void*)&task, sizeof(Task));

						CreateThread(NULL, 0, httpProc, NULL, 0, NULL);
					}
					else {

						numberOfAllErrors += prevNumberOfErrors - task.numberOfErrors;
					}
				}

				long long millisecondsSinceLatestSolution = GetTickCount64() - latestSolutionTime;

				if (millisecondsSinceLatestSolution > 15 * 60 * 1000) {

					numberOfChanges = 2;
				}
				else {

					numberOfChanges = 1;
				}

				char buffer[12];

				printf("\n--- Top 10 miners out of %d:                 [v0.3.0]\n", task.numberOfMiners);
				for (int i = 0; i < 10; i++) {

					printf(" #%2d   *   %10.10s...   *   %12s", i + 1, task.topMiners[i], number(task.topMinerScores[i], buffer));
					if (prevPrevTask.topMinerScores[i] == task.topMinerScores[i]) {

						printf("\n");
					}
					else {

						printf(" (increased by %s)\n", number(task.topMinerScores[i] - prevPrevTask.topMinerScores[i], buffer));
					}
				}
				printf("---\n");
				std::cout << InstructionSet::Brand() << std::endl;
				printf("   %s[AVX2]   %s[AVX512CD]   %s[AVX512ER]   %s[AVX512F]   %s[AVX512PF]\n", InstructionSet::AVX2() ? "+" : "-", InstructionSet::AVX512CD() ? "+" : "-", InstructionSet::AVX512ER() ? "+" : "-", InstructionSet::AVX512F() ? "+" : "-", InstructionSet::AVX512PF() ? "+" : "-");
				if (argc >= 4 && atoi(argv[3]) != 0) {

					std::wcout << L"   +[" << acc.description << "]" << std::endl;
				}
				printf("---\n");
				printf("You are #%d with %s found solutions", task.currentMinerPlace, number(task.currentMinerScore, buffer));
				if (prevPrevTask.currentMinerScore == task.currentMinerScore) {

					printf("\n");
				}
				else {

					printf(" (increased by %s)\n", number(task.currentMinerScore - prevPrevTask.currentMinerScore, buffer));
				}

				long long totalNumberOfIterations = 0;
				long long totalNumberOfGPUIterations = 0;
				for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]); i++) {

					totalNumberOfIterations += performanceCounters[i];
					totalNumberOfGPUIterations += performanceGPUCounters[i];
				}
				FILETIME currentTimestamp;
				GetSystemTimePreciseAsFileTime(&currentTimestamp);
				long long latestNumberOfIterations = numberOfIterations;
				long long latestNumberOfGPUIterations = numberOfGPUIterations;
				numberOfIterations = 0;
				numberOfGPUIterations = 0;
				ULARGE_INTEGER s, f;
				memcpy(&s, &performanceTimestamps[0], sizeof(ULARGE_INTEGER));
				memcpy(&f, &currentTimestamp, sizeof(ULARGE_INTEGER));
				totalNumberOfIterations += latestNumberOfIterations;
				totalNumberOfGPUIterations += latestNumberOfGPUIterations;
				for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1; i++) {

					CopyMemory(&performanceTimestamps[i], &performanceTimestamps[i + 1], sizeof(performanceTimestamps[0]));
					performanceCounters[i] = performanceCounters[i + 1];
					performanceGPUCounters[i] = performanceGPUCounters[i + 1];
				}
				CopyMemory(&performanceTimestamps[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1], &currentTimestamp, sizeof(performanceTimestamps[0]));
				performanceCounters[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1] = latestNumberOfIterations;
				performanceGPUCounters[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1] = latestNumberOfGPUIterations;
				printf("Your iteration rate on this hardware is %.3f (CPU) + %.3f (GPU) = %.3f iterations/s (%d/%d threads, %d changes)\n", totalNumberOfIterations * 10000000 / ((double)(f.QuadPart - s.QuadPart)), totalNumberOfGPUIterations * 10000000 / ((double)(f.QuadPart - s.QuadPart)), (totalNumberOfIterations + totalNumberOfGPUIterations) * 10000000 / ((double)(f.QuadPart - s.QuadPart)), numberOfThreads, systemInfo.dwNumberOfProcessors, numberOfChanges);

				double delta = ((double)(GetTickCount64() - launchTime)) / 1000;
				printf("Your error elimination rate on this hardware is ~%.3f errors/h (%s zero eliminations)\n", numberOfOwnErrors * 3600 / delta, number(numberOf0Elimitations, buffer));
				printf("Pool error elimination rate is ~%.3f errors/h\n", numberOfAllErrors * 3600 / delta);
				printf("");
				printf("---\n");

				if (justChanged) {

					CopyMemory((void*)&prevPrevTask, (void*)&prevTask, sizeof(Task));
					CopyMemory((void*)&prevTask, (void*)&task, sizeof(Task));
				}

				char buffer2[16];
				printf("There are %s errors left (decreased by %s) (at least %02d:%02d:%02d since latest solution)\n\n", number(task.numberOfErrors, buffer), number(prevPrevTask.numberOfErrors - task.numberOfErrors, buffer2), millisecondsSinceLatestSolution / 3600000, millisecondsSinceLatestSolution % 3600000 / 60000, millisecondsSinceLatestSolution % 60000 / 1000);

				break;
			}
		}

		Sleep(5000);
	}
}