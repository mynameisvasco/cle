#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int is_vowel(int c)
{
    int ac_vowels[40] = {
        0x61,
        0x65,
        0x69,
        0x6F,
        0x75,
        0x41,
        0x45,
        0x49,
        0x4F,
        0x55,
        0xC3A1,
        0xC3A0,
        0xC3A2,
        0xC3A3,
        0xC3A9,
        0xC3A8,
        0xC3AA,
        0xC3AD,
        0xC3AC,
        0xC3B3,
        0xC3B2,
        0xC3B4,
        0xC3B5,
        0xC3BA,
        0xC3B9,
        0xC381,
        0xC380,
        0xC382,
        0xC383,
        0xC389,
        0xC388,
        0xC38A,
        0xC38D,
        0xC38C,
        0xC393,
        0xC392,
        0xC394,
        0xC395,
        0xC39A,
        0xC399};

    for (int i = 0; i < 40; i++)
    {
        if (c == ac_vowels[i])
        {
            return 1;
        }
    }

    return 0;
}

int is_consonant(unsigned int c)
{
    int lower_cedilha = 0xC3A7;
    int upper_cedilha = 0xC387;
    return !is_vowel(c) && (((c >= 65 && c <= 90) || (c >= 97 && c <= 122)) || c == lower_cedilha || c == upper_cedilha);
}

int is_separator(unsigned int c)
{
    int dash = 0xE28093;
    int ellipsis = 0xE280A6;
    int doubleQuotationMarkLeft = 0xE2809C;
    int doubleQuotationMarkRight = 0xE2809D;
    return c == '[' || c == ']' || c == '(' || c == ')' || c == '-' || c == '"' || c == 0x20 || c == 0x9 || c == 0xA || c == 0xD || c == '.' || c == '?' || c == ';' || c == ':' || c == ',' || c == '!' || c == dash || c == ellipsis || c == doubleQuotationMarkLeft || c == doubleQuotationMarkRight;
}

int is_number(unsigned int c)
{
    return c >= 48 && c <= 57;
}

int read_utf8_char(FILE *file)
{
    unsigned char buffer[4] = {0, 0, 0, 0};
    unsigned int c = 0;
    fread(buffer, 1, 1, file);
    c = buffer[0];

    if ((buffer[0] >> 5) == 0b110)
    {
        fread(&buffer[1], 1, 1, file);
        c = (c << 8) | (buffer[1] & 0xff);
    }
    else if ((buffer[0] >> 4) == 0b1110)
    {
        fread(&buffer[1], 1, 2, file);
        c = (c << 8) | (buffer[1] & 0xff);
        c = (c << 8) | (buffer[2] & 0xff);
    }
    else if ((buffer[0] >> 3) == 0b11110)
    {
        fread(&buffer[1], 1, 3, file);
        c = (c << 8) | (buffer[1] & 0xff);
        c = (c << 8) | (buffer[2] & 0xff);
        c = (c << 8) | (buffer[2] & 0xff);
    }

    return c;
}

void process_file(const char *file_path)
{
    int is_in_word = 0;
    int words_number = 0;
    int words_vowel_start_number = 0;
    int words_consonant_ending_number = 0;
    FILE *file = fopen(file_path, "rb");

    if (file == NULL)
    {
        printf("File %s not found.\n", file_path);
        exit(-1);
    }

    int last_c = 0;

    while (1)
    {
        int c = read_utf8_char(file);

        if (c == '\0')
        {
            break;
        }
        else if (is_in_word && is_separator(c))
        {
            if (is_consonant(last_c))
            {
                words_consonant_ending_number++;
            }

            is_in_word = 0;
        }
        else if (!is_in_word && (is_vowel(c) || is_number(c) || is_consonant(c) || c == '_'))
        {
            if (is_vowel(c))
            {
                words_vowel_start_number++;
            }

            is_in_word = 1;
            words_number++;
        }

        if (is_separator(c) || is_vowel(c) || is_number(c) || is_consonant(c) || c == '_' || c == '\'')
        {
            last_c = c;
        }
    }

    printf("File: %s\n", file_path);
    printf("Number of words: %d\n", words_number);
    printf("Number of words starting with vowels: %d\n", words_vowel_start_number);
    printf("Number of words ending with consonants: %d\n", words_consonant_ending_number);
    fclose(file);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Wrong number of arguments\n, Usage: ./a.out test1.txt test2.txt ...\n");
        exit(-1);
    }

    clock_t begin = clock();

    for (int i = 1; i < argc; i++)
    {
        process_file(argv[i]);
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Elapsed time = %.3f s\n", time_spent);
}