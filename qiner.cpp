#include <intrin.h>
#include <stdio.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#define LIMIT 10000

__m256i aWords[19683][27];
__m256i bWords[(19683 + 255) / 256][27];
__m256i cWords[19683][(19683 + 255) / 256];

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

int main(int argc, char* argv[]) {

	if (argc < 2) {

		printf("qiner.exe <MyIdentity>\n");

		return 0;
	}

	__m256i flags[256];

	int i = 0;
	for (int j = 0; j < 8; j++) {

		int fragments[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

		for (fragments[j] = 1; fragments[j] != 0; fragments[j] <<= 1) {

			flags[i++] = _mm256_set_epi32(fragments[7], fragments[6], fragments[5], fragments[4], fragments[3], fragments[2], fragments[1], fragments[0]);
		}
	}

	for (int a = -9841; a <= 9841; a++) {

		int absoluteValue = a < 0 ? -a : a;
		for (int i = 0; i < 9; i++) {

			int remainder = absoluteValue % 3;
			absoluteValue = (absoluteValue + 1) / 3;
			aWords[a + 9841][i * 3] = _mm256_setzero_si256();
			aWords[a + 9841][i * 3 + 1] = _mm256_setzero_si256();
			aWords[a + 9841][i * 3 + 2] = _mm256_setzero_si256();
			aWords[a + 9841][((a < 0 && remainder != 0) ? (remainder ^ 3) : remainder) + i * 3] = _mm256_cmpeq_epi32(aWords[a + 9841][i * 3], aWords[a + 9841][i * 3]);
		}

		int b = -9841 - 1;
		for (int counter = 0; counter < (19683 + 255) / 256; counter++) {

			for (int i = 0; i < 27; i++) {

				bWords[counter][i] = _mm256_setzero_si256();
			}
			cWords[a + 9841][counter] = _mm256_setzero_si256();

			for (int f = 0; f < 256; f++) {

				if (++b > 9841) {

					b = -9841;
				}

				int c = (b == 0 ? 0 : a / b);

				int absoluteValueB = b < 0 ? -b : b;
				int absoluteValueC = c < 0 ? -c : c;
				for (int i = 0; i < 9; i++) {

					int remainderB = absoluteValueB % 3;
					absoluteValueB = (absoluteValueB + 1) / 3;
					bWords[counter][((b < 0 && remainderB != 0) ? (remainderB ^ 3) : remainderB) + i * 3] = _mm256_or_si256(bWords[counter][((b < 0 && remainderB != 0) ? (remainderB ^ 3) : remainderB) + i * 3], flags[f]);
				}
				if (absoluteValueC % 3 == 0) {

					cWords[a + 9841][counter] = _mm256_or_si256(cWords[a + 9841][counter], flags[f]);
				}
			}
		}
	}

	WSADATA wsaData;
	WSAStartup(MAKEWORD(2, 2), &wsaData);

	while (TRUE) {

		struct Task {

			int numberOfMiners;
			char topMiners[70][10];
			int topMinerScores[10];
			int currentMinerPlace;
			int currentMinerScore;
			int numberOfErrors;
			int links[LIMIT][3];

		} task;

		char getTask[71];
		getTask[0] = 0;
		CopyMemory(&getTask[1], argv[1], 70);
		if (exchange(getTask, sizeof(getTask), (char*)&task, sizeof(task)) != sizeof(task)) {

			printf("Failed to receive a task!\n");

			Sleep(5000);
		}
		else {

			printf("--- Top 10 miners out of %d:\n", task.numberOfMiners);
			for (int i = 0; i < 10; i++) {

				printf(" #%2d   *   %.10s...   *   %6d\n", i, task.topMiners[i], task.topMinerScores[i]);
			}
			printf("---\n");
			printf("You are #%d with %d found solutions\n", task.currentMinerPlace, task.currentMinerScore);
			printf("---\n");
			printf("There are %d errors left\n\n", task.numberOfErrors);

			FILETIME start;
			GetSystemTimePreciseAsFileTime(&start);

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

			int neuronLayers[LIMIT];
			for (int i = 0; i < 54; i++) {

				neuronLayers[i] = 0;
			}
			for (int i = 54; i < currentNeuronIndex; i++) {

				neuronLayers[i] = neuronLayers[links[i][0]];
				if (neuronLayers[links[i][1]] > neuronLayers[i]) {

					neuronLayers[i] = neuronLayers[links[i][1]];
				}
				if (neuronLayers[links[i][2]] > neuronLayers[i]) {

					neuronLayers[i] = neuronLayers[links[i][2]];
				}
				neuronLayers[i]++;
			}
			int numberOfLayers = neuronLayers[currentNeuronIndex - 1];

			int numberOfErrors = 0;

			for (int shiftedA = 0; numberOfErrors < task.numberOfErrors && shiftedA < 19683; shiftedA++) {

				__m256i values0[LIMIT];
				__m256i values1[LIMIT];

				memcpy(&values0[0], &aWords[shiftedA], sizeof(__m256i) * 27);
				memcpy(&values1[0], &aWords[shiftedA], sizeof(__m256i) * 27);

				for (int bChunkIndex = 0; bChunkIndex < (19683 + 255) / 256 - 1; ) {

					memcpy(&values0[27], &bWords[bChunkIndex], sizeof(__m256i) * 27);
					memcpy(&values1[27], &bWords[bChunkIndex + 1], sizeof(__m256i) * 27);

					int nextLink0 = links[54][0], nextLink1 = links[54][1], nextLink2 = links[54][2];
					for (int i = 54; i < currentNeuronIndex; ) {

						const __m256i tmp00 = _mm256_and_si256(values0[nextLink0], values0[nextLink1]);
						const __m256i tmp10 = _mm256_and_si256(values1[nextLink0], values1[nextLink1]);
						const __m256i tmp01 = _mm256_and_si256(tmp00, values0[nextLink2]);
						const __m256i tmp11 = _mm256_and_si256(tmp10, values1[nextLink2]);
						values0[i] = _mm256_xor_si256(tmp01, _mm256_cmpeq_epi32(tmp01, tmp01));
						values1[i++] = _mm256_xor_si256(tmp11, _mm256_cmpeq_epi32(tmp11, tmp11));
						nextLink0 = links[i][0];
						nextLink1 = links[i][1];
						nextLink2 = links[i][2];
					}

					long differences0[4], differences1[4];
					_mm256_storeu_si256((__m256i*) differences0, _mm256_xor_si256(values0[currentNeuronIndex - 1], cWords[shiftedA][bChunkIndex++]));
					_mm256_storeu_si256((__m256i*) differences1, _mm256_xor_si256(values1[currentNeuronIndex - 1], cWords[shiftedA][bChunkIndex++]));
					numberOfErrors += (int) (_mm_popcnt_u64(differences0[0]) + _mm_popcnt_u64(differences0[1]) + _mm_popcnt_u64(differences0[2]) + _mm_popcnt_u64(differences0[3])
						+ _mm_popcnt_u64(differences1[0]) + _mm_popcnt_u64(differences1[1]) + _mm_popcnt_u64(differences1[2]) + _mm_popcnt_u64(differences1[3]));
				}

				memcpy(&values0[27], &bWords[(19683 + 255) / 256 - 1], sizeof(__m256i) * 27);

				for (int i = 54; i < currentNeuronIndex; i++) {

					const __m256i tmp01 = _mm256_and_si256(_mm256_and_si256(values0[links[i][0]], values0[links[i][1]]), values0[links[i][2]]);
					values0[i] = _mm256_xor_si256(tmp01, _mm256_cmpeq_epi32(tmp01, tmp01));
				}

				long differences0[4];
				_mm256_storeu_si256((__m256i*) differences0, _mm256_xor_si256(values0[currentNeuronIndex - 1], cWords[shiftedA][(19683 + 255) / 256 - 1]));
				numberOfErrors += (int) (_mm_popcnt_u64(differences0[0]) + _mm_popcnt_u64(differences0[1]) + _mm_popcnt_u64(differences0[2]) + _mm_popcnt_u64(differences0[3]));
			}

			if (numberOfErrors < task.numberOfErrors) {

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
				CopyMemory(solution.currentMiner, argv[1], 70);
				solution.numberOfErrors = numberOfErrors;
				CopyMemory(solution.links, task.links, sizeof(solution.links));
				if (exchange((char*) &solution, sizeof(solution), NULL, 0) < 0) {

					printf("Failed to send a solution!\n");
				}
				else {

					printf("Managed to find a solution reducing number of errors to %d (%d neurons in %d layers) within %d ms\n\n", numberOfErrors, numberOfNeurons, numberOfLayers, (f.QuadPart - s.QuadPart) / 10000);
				}
			}
		}
	}
}