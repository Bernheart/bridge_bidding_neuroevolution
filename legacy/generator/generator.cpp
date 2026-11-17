#include <iostream>
#include <algorithm>
#include <random>
#include <array>
#include <sstream>
#include "./dds/include/dll.h"
#include "./dds/examples/hands.h"

constexpr unsigned int R2 = 0x0004;
constexpr unsigned int R3 = 0x0008;
constexpr unsigned int R4 = 0x0010;
constexpr unsigned int R5 = 0x0020;
constexpr unsigned int R6 = 0x0040;
constexpr unsigned int R7 = 0x0080;
constexpr unsigned int R8 = 0x0100;
constexpr unsigned int R9 = 0x0200;
constexpr unsigned int RT = 0x0400;
constexpr unsigned int RJ = 0x0800;
constexpr unsigned int RQ = 0x1000;
constexpr unsigned int RK = 0x2000;
constexpr unsigned int RA = 0x4000;

constexpr unsigned int cards[] = {RA, RK, RQ, RJ, RT, R9, R8, R7, R6, R5, R4, R3, R2};
constexpr int CARDS_IN_DECK = 52;
constexpr int HAND_SIZE = 13;
constexpr int HANDS = 4;
constexpr int TEST_SIZE = 32; // 32
constexpr int TEST_COUNT = 2; // 2
constexpr int BATCH_SIZE = 64; // 64
constexpr int N = 0, S = 2, E = 1, W = 3;
constexpr int SPADES = 0, HEARTS = 1, DIAMONDS = 2, CLUBS = 3, NOTRUMP = 4;
constexpr int SUITS = 5;
constexpr int LEVELS = 7;
constexpr int EV_SIZE = 1 + SUITS * LEVELS; // 36
constexpr int SIDES[2] = {N, S};
constexpr int MODE = -1; // No par calculation
int TRUMP_FILTER[SUITS] = {0}; // All

int score_table[5][7][2][14]; // [suit][level][vulnerable][tricks_taken]

// Define a type for holding the cards
using Deal = std::array<std::array<unsigned int, CARDS_IN_DECK>, HANDS>;
using Evaluation = std::array<std::array<int, EV_SIZE>, 2>;

int calculate_bridge_score(int suit, int level, bool vulnerable, int tricks_taken) {
    const int contract_tricks = 6 + (++level);  // Promote level inline
    if (tricks_taken < contract_tricks) {
        const int undertricks = contract_tricks - tricks_taken;
        return -(vulnerable ? 100 : 50) * undertricks;
    }

    int trickScore = 0;
    switch (suit) {
        case CLUBS: case DIAMONDS: trickScore = 20; break;
        case HEARTS: case SPADES:  trickScore = 30; break;
        case NOTRUMP:              trickScore = 30; break;
        default:
            std::cerr << "Invalid suit: " << suit << std::endl;
            return -1;
    }

    int base = level * trickScore + (suit == NOTRUMP ? 10 : 0);
    int score = base;

    score += (base >= 100) ? (vulnerable ? 500 : 300) : 50;

    if (level == 6) score += vulnerable ? 750 : 500;
    else if (level == 7) score += vulnerable ? 1500 : 1000;

    score += (tricks_taken - contract_tricks) * trickScore;

    return score;
}

void arr_to_ddTableDeal(Deal& deal_bitmap, ddTableDeal &deal) {
    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < CARDS_IN_DECK; ++j)
    //         std::cout << deal[i][j] << " ";
    //     std::cout << "\n";
    // }
    // std::cout << "\n";

    for (int i : SIDES)
    {
        for (int s = 0; s < DDS_SUITS; s++)
            deal.cards[i][s] = 0;
        for (int j = 0; j < CARDS_IN_DECK; ++j)
        {
            if (deal_bitmap[i][j] == 1)
            {
                // std::cout << "BeforeXXX: " <<std::hex << deal_table.cards[i][j / HAND_SIZE] << " ";
                deal.cards[i][j / HAND_SIZE] |= cards[j % HAND_SIZE];
                // std::cout << "AfterXXX: " <<std::hex << deal_table.cards[i][j / HAND_SIZE] << " ";
                // std::cout << "Hand: " << i << " Suit:" << j / HAND_SIZE << " Card: " << 14 - (j % HAND_SIZE) << " " <<std::hex<< cards[j % HAND_SIZE] << std::endl;
            }
                // deal_table.cards[i][j / HAND_SIZE] |= cards[j % HAND_SIZE];
        }
    }

    // for (int h = 0; h < DDS_HANDS; h++)
    // {
    //     for (int s = 0; s < DDS_SUITS; s++)
    //     {
    //         std::cout << "Before: " <<std::hex << deal_table.cards[s][h] << " ";
    //         deal_table.cards[h][s] = holdings[0][s][h];
    //         std::cout << "After: " <<std::hex << deal_table.cards[s][h] << " ";
    //     }
    //     std::cout << std::endl << std::hex << deal_table.cards[h];
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
            
}

// void arr_to_ddTableDeals(Deal test[TEST_SIZE], ddTableDeals& deals) {
//     deals.noOfTables = TEST_SIZE;

//     for (int i = 0; i < TEST_SIZE; ++i)
//         arr_to_ddTableDeal(test[i], deals.deals[i]);
// }

void generate_tests(ddTableDeal& deal, ddTableDeals& deals, std::vector<int>& available_positions, std::mt19937& rng)
{
    for(int i = 0; i < TEST_SIZE; i++)
    {
        std::fill(deal.cards[E], deal.cards[E] + DDS_SUITS, 0);
        std::fill(deal.cards[W], deal.cards[W] + DDS_SUITS, 0);
        std::shuffle(available_positions.begin(), available_positions.end(), rng);

        for (int i = 0; i < HAND_SIZE; i++) {
            deal.cards[E][available_positions[i] / HAND_SIZE] |= cards[available_positions[i] % HAND_SIZE];
            deal.cards[W][available_positions[i + HAND_SIZE]/HAND_SIZE] |= 
                cards[available_positions[i+HAND_SIZE] % HAND_SIZE];
        }
        // Copy deal into tests[i]
        deals.deals[i] = deal;
    }
}

void fill_ev(Evaluation& ev, ddTablesRes& tableRes, bool& vul) {
    for (int test_no = 0; test_no < TEST_SIZE; test_no++)
    {
        for (int sl = 0; sl < SUITS * LEVELS; sl++)
        {
            int s = sl / LEVELS;
            int l = sl % LEVELS;
            int i = 1 + s*LEVELS + l;
            ev[0][i] += score_table[s][l][vul][tableRes.results[test_no].resTable[s][N]];
            ev[1][i] += score_table[s][l][vul][tableRes.results[test_no].resTable[s][S]];
        }
    }
}

std::string generate_deal_output(std::mt19937& rng,
    Evaluation &ev, ddTablesRes& tableRes,
    ddTableDeals& deals, ddTableDeal& deal,
    std::vector<int>& available_positions, 
    Deal& deal_bitmap, bool& vul) 
{
    // fill available positions
    available_positions.clear();
    for (int i = 0; i < CARDS_IN_DECK; i++)
        available_positions.push_back(i);
    
    // shuffle for random deal
    std::shuffle(available_positions.begin(), available_positions.end(), rng);

    // fill bitmap with 0
    std::fill(deal_bitmap[N].begin(), deal_bitmap[N].end(), 0);
    std::fill(deal_bitmap[S].begin(), deal_bitmap[S].end(), 0);

    // fill bitmap for N and S with random cards
    for (int i = 0; i < HAND_SIZE; i++) 
    {
        deal_bitmap[N][available_positions[i]] = 1;
        deal_bitmap[S][available_positions[i+HAND_SIZE]] = 1;
    }

    // leave the rest of the cards for E and W
    available_positions = std::vector<int>(available_positions.begin() + HAND_SIZE*2, available_positions.end());

    // change the bitmap to ddTableDeal
    arr_to_ddTableDeal(deal_bitmap, deal);

    // fill ev with 0
    std::fill(ev[0].begin(), ev[0].end(), 0);
    std::fill(ev[1].begin(), ev[1].end(), 0);
    
    for(int test_batch = 0; test_batch < TEST_COUNT; test_batch++) 
    {
        
        generate_tests(deal, deals, available_positions, rng);
        // arr_to_ddTableDeals(tests, deals);
        
        
        int res = CalcAllTables(&deals, MODE, TRUMP_FILTER, &tableRes, nullptr);
        
        if (res != RETURN_NO_FAULT)
        {
            char line[80];
            ErrorMessage(res, line);
            printf("DDS error: %s\n", line);
        }

        // for (int test_no = 0; test_no < TEST_SIZE; test_no++)
        // {
        //     char line[80];
        //     PrintHand(line, deals.deals[test_no].cards);
    
        //     PrintTable(&tableRes.results[test_no]);
        // }

        
        fill_ev(ev, tableRes, vul);
    }

    // for (int i = 0; i < EV_SIZE; i++)
    // {
    //     std::cout << "Suit: " << (i - 1) / LEVELS << " Level:" << (i - 1) % LEVELS + 1  << " "<< ev_N[i] << " " << ev_N[i] / (TEST_SIZE * TEST_COUNT) << "\n";
    //     // std::cout << ev_S[i] << " " << ev_S[i] / (TEST_SIZE * TEST_COUNT) << "; ";
    // }
    // std::cout << std::endl;
    std::stringstream ss;

    for (int i : SIDES) {
        for (int j = 0; j < CARDS_IN_DECK; j++) {
            ss << deal_bitmap[i][j] << ",";
        }
        ss << vul << "\n";
    }

    for (int i : SIDES) {
        ss << 0; // pass
        for (int s = 0; s < SUITS; s++) {
            int temp_s = s < 4 ? 3 - s : s;
            for (int l = 0; l < LEVELS; l++) {
                ss << "," << (ev[i / 2][1 + temp_s * LEVELS + l] / (TEST_SIZE * TEST_COUNT));
            }
        }
        ss << "\n";
    }

    return ss.str();
}

// std::mutex cout_mutex;  // To avoid jumbled output from multiple threads

// void generator_thread(int thread_id, int count_per_thread) {
//     std::mt19937 rng(std::random_device{}()); // Local RNG per thread

//     for (int i = 0; i < count_per_thread; ++i)
//     {
//         std::string deal_output = generate_deal_output(rng);
//         // Thread-safe console output (optional)
//         {
//             std::lock_guard<std::mutex> lock(cout_mutex);
//             std::cout << deal_output;
//         }
//     }
// }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <seed>" << std::endl;
        return 1;
    }
    

    // Precompute once at the start
    for (int suit = 0; suit < SUITS; ++suit)
        for (int level = 0; level < LEVELS; ++level)
            for (int vul = 0; vul <= 1; ++vul)
                for (int tricks = 0; tricks <= 13; ++tricks)
                    score_table[suit][level][vul][tricks] = calculate_bridge_score(suit, level, vul, tricks);

    // int num_threads = std::thread::hardware_concurrency();  // e.g. 8
    // int deals_per_thread = BATCH_SIZE / num_threads;
    // int leftover = BATCH_SIZE % num_threads;
    // std::cout << "Number of threads: " << num_threads << "\n";
    SetMaxThreads(0);
    
    // std::vector<std::thread> threads;
    
    // for (int i = 0; i < num_threads; ++i) {
    //     int count = deals_per_thread + (i < leftover ? 1 : 0);  // Distribute leftovers
    //     threads.emplace_back(generator_thread, i, count);
    // }
    int seed = std::stoi(argv[1]);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1);

    Evaluation ev;
    ddTablesRes tableRes;
    ddTableDeals deals;
    deals.noOfTables = TEST_SIZE;
    ddTableDeal deal;
    std::vector<int> available_positions;
    Deal deal_bitmap;

    for (int batch = 0; batch < BATCH_SIZE; ++batch)
    {
        bool vul = dist(rng);
        std::string deal_output = generate_deal_output(rng, ev, tableRes, deals, deal, available_positions, deal_bitmap, vul);
        std::cout << deal_output;
    }
    // // Join all threads
    // for (auto& t : threads) t.join();


    // int seed = std::stoi(argv[1]);
    // std::mt19937 rng(seed);
    // for (int i = 0; i < BATCH_SIZE; ++i)
    // {
    //     // std::cout << "Batch: " << i << std::endl;
    //     generator(rng);
    //     if (i != BATCH_SIZE - 1)
    //         std::cout << "\n";
    // }

    return 0;
}