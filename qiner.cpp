#include <array>
#include <bitset>
#include <intrin.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <ws2tcpip.h>
#include <winsock2.h>

#define LIMIT 10000

__declspec(align(32)) const __m256i allOnes = _mm256_set1_epi32(0xFFFFFFFF);

long long launchTime = GetTickCount64() - 1;
volatile long long numberOfIterations, numberOfXIterations, numberOfOwnErrors = 0, numberOfOwnXErrors = 0, numberOf0Elimitations = 0, numberOf0XElimitations = 0;
long long numberOfAllErrors = 0;

FILETIME performanceTimestamps[12];
long long performanceCounters[12];
long long performanceXCounters[12];

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

		int prevNumberOfNeurons = 0;

		{
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

					prevNumberOfNeurons++;
				}
			}
		}

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

		if (lpParameter && numberOfNeurons < prevNumberOfNeurons) {

			InterlockedIncrement64(&numberOfXIterations);
		}
		else {

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

			if (lpParameter) {

				InterlockedIncrement64(&numberOfXIterations);
			}
			else {

				InterlockedIncrement64(&numberOfIterations);
			}

			BOOL improved = numberOfErrors < task.numberOfErrors ? TRUE : FALSE;
			if (numberOfErrors <= task.numberOfErrors) {

				if (numberOfErrors < task.numberOfErrors) {

					if (lpParameter) {

						InterlockedAdd64(&numberOfOwnXErrors, task.numberOfErrors - numberOfErrors);
					}
					else {

						InterlockedAdd64(&numberOfOwnErrors, task.numberOfErrors - numberOfErrors);
					}
				}
				else {

					if (lpParameter) {

						InterlockedIncrement64(&numberOf0XElimitations);
					}
					else {

						InterlockedIncrement64(&numberOf0Elimitations);
					}
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
							printf("AVX2: Found a solution reducing number of errors by %s (%s neurons)\n\n", number(task.numberOfErrors - numberOfErrors, buffer), number(numberOfNeurons, buffer2));
						}
					}

					Sleep(5000);

				} while (improved && numberOfErrors < ::task.numberOfErrors);
			}
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

		int prevNumberOfNeurons = 0;

		{
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

					prevNumberOfNeurons++;
				}
			}
		}

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

		if (lpParameter && numberOfNeurons < prevNumberOfNeurons) {

			InterlockedIncrement64(&numberOfXIterations);
		}
		else {

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

					long long differences0[4], differences1[4], differences2[4];
					*((__m256i*)differences0) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][0], diffMask), cWords[a][b][0]);
					*((__m256i*)differences1) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][1], diffMask), cWords[a][b][1]);
					*((__m256i*)differences2) = _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][2], diffMask), cWords[a][b][2]);
					numberOfErrors += (int)(_mm_popcnt_u64(differences0[0]) + _mm_popcnt_u64(differences0[1]) + _mm_popcnt_u64(differences0[2]) + _mm_popcnt_u64(differences0[3])
						+ _mm_popcnt_u64(differences1[0]) + _mm_popcnt_u64(differences1[1]) + _mm_popcnt_u64(differences1[2]) + _mm_popcnt_u64(differences1[3])
						+ _mm_popcnt_u64(differences2[0]) + _mm_popcnt_u64(differences2[1]) + _mm_popcnt_u64(differences2[2]) + _mm_popcnt_u64(differences2[3]));
				}
			}

			if (lpParameter) {

				InterlockedIncrement64(&numberOfXIterations);
			}
			else {

				InterlockedIncrement64(&numberOfIterations);
			}

			BOOL improved = numberOfErrors < task.numberOfErrors ? TRUE : FALSE;
			if (numberOfErrors <= task.numberOfErrors) {

				if (numberOfErrors < task.numberOfErrors) {

					if (lpParameter) {

						InterlockedAdd64(&numberOfOwnXErrors, task.numberOfErrors - numberOfErrors);
					}
					else {

						InterlockedAdd64(&numberOfOwnErrors, task.numberOfErrors - numberOfErrors);
					}
				}
				else {

					if (lpParameter) {

						InterlockedIncrement64(&numberOf0XElimitations);
					}
					else {

						InterlockedIncrement64(&numberOf0Elimitations);
					}
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
							printf("AVX-512: Found a solution reducing number of errors by %s (%s neurons)\n\n", number(task.numberOfErrors - numberOfErrors, buffer), number(numberOfNeurons, buffer2));
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

		printf("qiner.exe <MyIdentity> <NumberOfClassicalThreads> <NumberOfExperimentalThreads>\n");

		return 0;
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
	int numberOfXThreads = (argc >= 4) ? atoi(argv[3]) : 0;

	for (int i = 0; i < numberOfThreads; i++) {

		CreateThread(NULL, 2 * 1024 * 1024, InstructionSet::AVX512F() ? miningProcAVX512 : miningProc, NULL, 0, NULL);
	}
	for (int i = 0; i < numberOfXThreads; i++) {

		CreateThread(NULL, 2 * 1024 * 1024, InstructionSet::AVX512F() ? miningProcAVX512 : miningProc, (LPVOID)1, 0, NULL);
	}

	long long latestSolutionTime = GetTickCount64();

	for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]); i++) {

		GetSystemTimePreciseAsFileTime(&performanceTimestamps[i]);
		performanceCounters[i] = 0;
		performanceXCounters[i] = 0;
	}
	numberOfIterations = 0;
	numberOfXIterations = 0;

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

				printf("\n--- Top 10 miners out of %d:                 [v0.3.1]\n", task.numberOfMiners);
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
				printf("---\n");
				printf("You are #%d with %s found solutions", task.currentMinerPlace, number(task.currentMinerScore, buffer));
				if (prevPrevTask.currentMinerScore == task.currentMinerScore) {

					printf("\n");
				}
				else {

					printf(" (increased by %s)\n", number(task.currentMinerScore - prevPrevTask.currentMinerScore, buffer));
				}

				long long totalNumberOfIterations = 0;
				long long totalNumberOfXIterations = 0;
				for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]); i++) {

					totalNumberOfIterations += performanceCounters[i];
					totalNumberOfXIterations += performanceXCounters[i];
				}
				FILETIME currentTimestamp;
				GetSystemTimePreciseAsFileTime(&currentTimestamp);
				long long latestNumberOfIterations = numberOfIterations;
				long long latestNumberOfXIterations = numberOfXIterations;
				numberOfIterations = 0;
				numberOfXIterations = 0;
				ULARGE_INTEGER s, f;
				memcpy(&s, &performanceTimestamps[0], sizeof(ULARGE_INTEGER));
				memcpy(&f, &currentTimestamp, sizeof(ULARGE_INTEGER));
				totalNumberOfIterations += latestNumberOfIterations;
				totalNumberOfXIterations += latestNumberOfXIterations;
				for (int i = 0; i < sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1; i++) {

					CopyMemory(&performanceTimestamps[i], &performanceTimestamps[i + 1], sizeof(performanceTimestamps[0]));
					performanceCounters[i] = performanceCounters[i + 1];
					performanceXCounters[i] = performanceXCounters[i + 1];
				}
				CopyMemory(&performanceTimestamps[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1], &currentTimestamp, sizeof(performanceTimestamps[0]));
				performanceCounters[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1] = latestNumberOfIterations;
				performanceXCounters[sizeof(performanceCounters) / sizeof(performanceCounters[0]) - 1] = latestNumberOfXIterations;
				printf("Your iteration rate is %.3f (classical) + %.3f (experimental) = %.3f iterations/s ([%d+%d]/%d threads, %d changes)\n", totalNumberOfIterations * 10000000 / ((double)(f.QuadPart - s.QuadPart)), totalNumberOfXIterations * 10000000 / ((double)(f.QuadPart - s.QuadPart)), (totalNumberOfIterations + totalNumberOfXIterations) * 10000000 / ((double)(f.QuadPart - s.QuadPart)), numberOfThreads, numberOfXThreads, systemInfo.dwNumberOfProcessors, numberOfChanges);

				double delta = ((double)(GetTickCount64() - launchTime)) / 1000;
				printf("Your error elimination rate is ~%.3f errors/h (%s zero eliminations) @ Classical\n", numberOfOwnErrors * 3600 / delta, number(numberOf0Elimitations, buffer));
				printf("Your error elimination rate is ~%.3f errors/h (%s zero eliminations) @ Experimental\n", numberOfOwnXErrors * 3600 / delta, number(numberOf0XElimitations, buffer));
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