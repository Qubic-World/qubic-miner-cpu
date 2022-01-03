#include <intrin.h>
#include <stdio.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#define LIMIT 10000

const __m256i allOnes = _mm256_set1_epi32(0xFFFFFFFF);

long long launchTime = 0;
volatile long long numberOfIterations = 0, numberOfOwnErrors = 0, numberOf0Elimitations = 0;
int numberOfAllErrors = 0;

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

} task, prevTask;

__m256i diffMask = _mm256_setzero_si256();

auto inputOutputs = new __m256i[19683 * 19683 / 243 / 3][27 + (27 + 27 + 27) + (1 + 1 + 1)];

int exchange(char* dataToSend, int dataToSendSize, char* receivedDataBuffer, int receivedDataBufferSize) {

	SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	sockaddr_in addr;
	ZeroMemory(&addr, sizeof(addr));
	addr.sin_family = AF_INET;
	inet_pton(AF_INET, "88.99.67.51", &addr.sin_addr.s_addr);
	addr.sin_port = htons(21841);
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

		Sleep(1000);
	}
	Sleep(1000);

	if (launchTime == 0) {

		launchTime = GetTickCount64() - 1;
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

		__m256i values[LIMIT][3];
		for (int inputOutputIndex = 0; numberOfErrors <= task.numberOfErrors && inputOutputIndex < 19683 * 19683 / 243 / 3; inputOutputIndex++) {

			for (int i = 0; i < 27; i++) {

				values[i][2] = values[i][1] = values[i][0] = inputOutputs[inputOutputIndex][i];
			}
			for (int i = 0; i < 27; i++) {

				values[27 + i][0] = inputOutputs[inputOutputIndex][27 + i * 3];
				values[27 + i][1] = inputOutputs[inputOutputIndex][27 + i * 3 + 1];
				values[27 + i][2] = inputOutputs[inputOutputIndex][27 + i * 3 + 2];
			}

			for (int i = 54; i < currentNeuronIndex; i++) {

				//values[i][0] = _mm256_ternarylogic_epi32(values[links[i][0]][0], values[links[i][1]][0], values[links[i][2]][0], 0x7F);
				//values[i][1] = _mm256_ternarylogic_epi32(values[links[i][0]][1], values[links[i][1]][1], values[links[i][2]][1], 0x7F);
				//values[i][2] = _mm256_ternarylogic_epi32(values[links[i][0]][2], values[links[i][1]][2], values[links[i][2]][2], 0x7F);

				const __m256i tmp00 = _mm256_and_si256(values[links[i][0]][0], values[links[i][1]][0]);
				const __m256i tmp10 = _mm256_and_si256(values[links[i][0]][1], values[links[i][1]][1]);
				const __m256i tmp20 = _mm256_and_si256(values[links[i][0]][2], values[links[i][1]][2]);
				const __m256i tmp01 = _mm256_and_si256(tmp00, values[links[i][2]][0]);
				const __m256i tmp11 = _mm256_and_si256(tmp10, values[links[i][2]][1]);
				const __m256i tmp21 = _mm256_and_si256(tmp20, values[links[i][2]][2]);
				values[i][0] = _mm256_xor_si256(tmp01, allOnes);
				values[i][1] = _mm256_xor_si256(tmp11, allOnes);
				values[i][2] = _mm256_xor_si256(tmp21, allOnes);
			}

			long long differences0[4], differences1[4], differences2[4];
			_mm256_storeu_si256((__m256i*)differences0, _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][0], diffMask), inputOutputs[inputOutputIndex][27 + (27 + 27 + 27)]));
			_mm256_storeu_si256((__m256i*)differences1, _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][1], diffMask), inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1]));
			_mm256_storeu_si256((__m256i*)differences2, _mm256_xor_si256(_mm256_and_si256(values[currentNeuronIndex - 1][2], diffMask), inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1 + 1]));
			numberOfErrors += (int)(_mm_popcnt_u64(differences0[0]) + _mm_popcnt_u64(differences0[1]) + _mm_popcnt_u64(differences0[2]) + _mm_popcnt_u64(differences0[3])
				+ _mm_popcnt_u64(differences1[0]) + _mm_popcnt_u64(differences1[1]) + _mm_popcnt_u64(differences1[2]) + _mm_popcnt_u64(differences1[3])
				+ _mm_popcnt_u64(differences2[0]) + _mm_popcnt_u64(differences2[1]) + _mm_popcnt_u64(differences2[2]) + _mm_popcnt_u64(differences2[3]));
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

						printf("Managed to find a solution reducing number of errors by %d (%d neurons) within %llu ms\n\n", (task.numberOfErrors - numberOfErrors), numberOfNeurons, (f.QuadPart - s.QuadPart) / 10000);
					}
				}

				Sleep(5000);

			} while (improved && numberOfErrors < ::task.numberOfErrors);
		}
	}
}

int main(int argc, char* argv[]) {

	if (argc < 3) {

		printf("qiner.exe <MyIdentity> <NumberOfMiningThreads> <NumberOfChanges>\n");

		return 0;
	}

	if (argc >= 4) {

		numberOfChanges = atoi(argv[3]);
	}

	__m256i flags[243];

	int i = 0;
	for (int j = 0; j < 8; j++) {

		int fragments[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

		for (fragments[j] = 1; fragments[j] != 0; fragments[j] <<= 1) {

			flags[i] = _mm256_set_epi32(fragments[7], fragments[6], fragments[5], fragments[4], fragments[3], fragments[2], fragments[1], fragments[0]);
			diffMask = _mm256_or_si256(diffMask, flags[i]);

			if (++i == 243) {

				break;
			}
		}
	}

	int a = -9841, b = -9841;
	ZeroMemory(inputOutputs, sizeof(inputOutputs));
	for (int inputOutputIndex = 0; inputOutputIndex < 19683 * 19683 / 243 / 3; inputOutputIndex++) {

		for (int i = 0; i < 243; i++) {

			int c0 = (b == 0 ? 0 : a / b);
			int c1 = ((b + 1) == 0 ? 0 : a / (b + 1));
			int c2 = ((b + 2) == 0 ? 0 : a / (b + 2));

			int absoluteValueA = a < 0 ? -a : a;
			int absoluteValueB0 = b < 0 ? -b : b;
			int absoluteValueB1 = (b + 1) < 0 ? -(b + 1) : (b + 1);
			int absoluteValueB2 = (b + 2) < 0 ? -(b + 2) : (b + 2);
			int absoluteValueC0 = c0 < 0 ? -c0 : c0;
			int absoluteValueC1 = c1 < 0 ? -c1 : c1;
			int absoluteValueC2 = c2 < 0 ? -c2 : c2;
			for (int j = 0; j < 9; j++) {

				int remainderA = absoluteValueA % 3;
				int remainderB0 = absoluteValueB0 % 3;
				int remainderB1 = absoluteValueB1 % 3;
				int remainderB2 = absoluteValueB2 % 3;
				absoluteValueA = (absoluteValueA + 1) / 3;
				absoluteValueB0 = (absoluteValueB0 + 1) / 3;
				absoluteValueB1 = (absoluteValueB1 + 1) / 3;
				absoluteValueB2 = (absoluteValueB2 + 1) / 3;
				inputOutputs[inputOutputIndex][((a < 0 && remainderA != 0) ? (remainderA ^ 3) : remainderA) + j * 3] = _mm256_or_si256(inputOutputs[inputOutputIndex][((a < 0 && remainderA != 0) ? (remainderA ^ 3) : remainderA) + j * 3], flags[i]);
				inputOutputs[inputOutputIndex][27 + (((b < 0 && remainderB0 != 0) ? (remainderB0 ^ 3) : remainderB0) + j * 3) * 3] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + (((b < 0 && remainderB0 != 0) ? (remainderB0 ^ 3) : remainderB0) + j * 3) * 3], flags[i]);
				inputOutputs[inputOutputIndex][27 + ((((b + 1) < 0 && remainderB1 != 0) ? (remainderB1 ^ 3) : remainderB1) + j * 3) * 3 + 1] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + ((((b + 1) < 0 && remainderB1 != 0) ? (remainderB1 ^ 3) : remainderB1) + j * 3) * 3 + 1], flags[i]);
				inputOutputs[inputOutputIndex][27 + ((((b + 2) < 0 && remainderB2 != 0) ? (remainderB2 ^ 3) : remainderB2) + j * 3) * 3 + 2] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + ((((b + 2) < 0 && remainderB2 != 0) ? (remainderB2 ^ 3) : remainderB2) + j * 3) * 3 + 2], flags[i]);
			}
			if (absoluteValueC0 % 3 == 0) {

				inputOutputs[inputOutputIndex][27 + (27 + 27 + 27)] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + (27 + 27 + 27)], flags[i]);
			}
			if (absoluteValueC1 % 3 == 0) {

				inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1], flags[i]);
			}
			if (absoluteValueC2 % 3 == 0) {

				inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1 + 1] = _mm256_or_si256(inputOutputs[inputOutputIndex][27 + (27 + 27 + 27) + 1 + 1], flags[i]);
			}

			b += 3;
		}

		if (b > 9841) {

			b = -9841;

			a++;
		}
	}

	CopyMemory(miner, argv[1], 70);

	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);

	task.numberOfErrors = 0x7FFFFFFF;

	for (int i = 0; i < atoi(argv[2]); i++) {

		CreateThread(NULL, 2 * 1024 * 1024, miningProc, NULL, 0, NULL);
	}

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

					justChanged = TRUE;

					if (prevNumberOfErrors == 0x7FFFFFFF) {

						CopyMemory((void*)&prevTask, (void*)&task, sizeof(Task));
					}
					else {

						numberOfAllErrors += prevNumberOfErrors - task.numberOfErrors;
					}
				}

				printf("--- Top 10 miners out of %d:            [v0.1.3]\n", task.numberOfMiners);
				for (int i = 0; i < 10; i++) {

					printf(" #%2d   *   %.10s...   *   %8d", i + 1, task.topMiners[i], task.topMinerScores[i]);
					if (prevTask.topMinerScores[i] == task.topMinerScores[i]) {

						printf("\n");
					}
					else {

						printf(" (increased by %d)\n", task.topMinerScores[i] - prevTask.topMinerScores[i]);
					}
				}
				printf("---\n");
				printf("You are #%d with %d found solutions", task.currentMinerPlace, task.currentMinerScore);
				if (prevTask.currentMinerScore == task.currentMinerScore) {

					printf("\n");
				}
				else {

					printf(" (increased by %d)\n", task.currentMinerScore - prevTask.currentMinerScore);
				}
				double delta = ((double)(GetTickCount64() - launchTime)) / 1000;
				printf("Your iteration rate on this hardware is %f iterations/s (%d changes)\n", numberOfIterations / delta, numberOfChanges);
				printf("Your error elimination rate on this hardware is %f errors/h (%llu zero eliminations)\n", numberOfOwnErrors * 3600 / delta, numberOf0Elimitations);
				printf("Pool error elimination rate is %f errors/h\n", numberOfAllErrors * 3600 / delta);
				printf("");
				printf("---\n");
				printf("There are %d errors left\n\n", task.numberOfErrors);

				if (justChanged) {

					CopyMemory((void*)&prevTask, (void*)&task, sizeof(Task));
				}

				break;
			}
		}

		Sleep(5000);
	}
}