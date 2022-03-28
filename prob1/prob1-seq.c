#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int are_utf8_chars_equal(char *c1, char *c2)
{
    int i = 0;

    while (i < 4)
    {
        if (*(c1 + i) != *(c2 + i) && *(c1 + i) != 0 && *(c2 + i) != 0)
        {
            return 0;
        }

        i++;
    }

    return 1;
}

int is_vowel(char *c)
{
    char ac_vowels[40][4] = {
        {0x61, 0x00, 0x00, 0x00},
        {0x65, 0x00, 0x00, 0x00},
        {0x69, 0x00, 0x00, 0x00},
        {0x6F, 0x00, 0x00, 0x00},
        {0x75, 0x00, 0x00, 0x00},
        {0x41, 0x00, 0x00, 0x00},
        {0x45, 0x00, 0x00, 0x00},
        {0x49, 0x00, 0x00, 0x00},
        {0x4F, 0x00, 0x00, 0x00},
        {0x55, 0x00, 0x00, 0x00},
        {0xC3, 0xA1, 0x00, 0x00},
        {0xC3, 0xA0, 0x00, 0x00},
        {0xC3, 0xA2, 0x00, 0x00},
        {0xC3, 0xA3, 0x00, 0x00},
        {0xC3, 0xA9, 0x00, 0x00},
        {0xC3, 0xA8, 0x00, 0x00},
        {0xC3, 0xAA, 0x00, 0x00},
        {0xC3, 0xAD, 0x00, 0x00},
        {0xC3, 0xAC, 0x00, 0x00},
        {0xC3, 0xB3, 0x00, 0x00},
        {0xC3, 0xB2, 0x00, 0x00},
        {0xC3, 0xB4, 0x00, 0x00},
        {0xC3, 0xB5, 0x00, 0x00},
        {0xC3, 0xBA, 0x00, 0x00},
        {0xC3, 0xB9, 0x00, 0x00},
        {0xC3, 0x81, 0x00, 0x00},
        {0xC3, 0x80, 0x00, 0x00},
        {0xC3, 0x82, 0x00, 0x00},
        {0xC3, 0x83, 0x00, 0x00},
        {0xC3, 0x89, 0x00, 0x00},
        {0xC3, 0x88, 0x00, 0x00},
        {0xC3, 0x8A, 0x00, 0x00},
        {0xC3, 0x8D, 0x00, 0x00},
        {0xC3, 0x8C, 0x00, 0x00},
        {0xC3, 0x93, 0x00, 0x00},
        {0xC3, 0x92, 0x00, 0x00},
        {0xC3, 0x94, 0x00, 0x00},
        {0xC3, 0x95, 0x00, 0x00},
        {0xC3, 0x9A, 0x00, 0x00},
        {0xC3, 0x99, 0x00, 0x00}};

    for (int i = 0; i < 40; i++)
    {
        if (are_utf8_chars_equal(c, ac_vowels[i]))
            return 1;
    }

    return 0;
}

int is_consonant(char *c)
{
    char lower_cedilha[4] = {0xC3, 0xA7, 0x00, 0x00};
    char upper_cedilha[4] = {0xC3, 0x87, 0x00, 0x00};
    return !is_vowel(c) && (((*c >= 65 && *c <= 90) || (*c >= 97 && *c <= 122)) || are_utf8_chars_equal(c, lower_cedilha) || are_utf8_chars_equal(c, upper_cedilha));
}

int is_separator(char *c)
{
    char dash[4] = {0xE2, 0x80, 0x93, 0x00};
    char ellipsis[4] = {0xE2, 0x80, 0xA6, 0x00};
    char doubleQuotationMarkLeft[4] = {0xE2, 0x80, 0x9C, 0x00};
    char doubleQuotationMarkRight[4] = {0xE2, 0x80, 0x9D, 0x00};
    return *c == '[' || *c == ']' || *c == '(' || *c == ')' || *c == '-' || *c == '"' || *c == 0x20 || *c == 0x9 || *c == 0xA || *c == 0xD || *c == '.' || *c == '?' || *c == ';' || *c == ':' || *c == ',' || *c == '!' || are_utf8_chars_equal(c, dash) || are_utf8_chars_equal(c, ellipsis) || are_utf8_chars_equal(c, doubleQuotationMarkLeft) || are_utf8_chars_equal(c, doubleQuotationMarkRight);
}

int is_number(char *c)
{
    return *c >= 48 && *c <= 57;
}

char *read_utf8_char(FILE *file)
{
    char *buffer = malloc(sizeof(char) * 4);
    buffer[0] = 0;
    fread(buffer, 1, 1, file);
    buffer[1] = 0;
    buffer[2] = 0;
    buffer[3] = 0;

    if (buffer[0] >= -64 && buffer[0] <= -33)
        fread(buffer + 1, 1, 1, file);
    else if (buffer[0] >= -32 && buffer[0] <= -17)
        fread(buffer + 1, 1, 2, file);
    else if (buffer[0] >= -16 && buffer[0] <= -9)
        fread(buffer + 1, 1, 3, file);

    return buffer;
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

    char *last_c = malloc(sizeof(char) * 4);

    while (1)
    {
        char *c = read_utf8_char(file);

        if (*c == '\0')
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
        else if (!is_in_word && (is_vowel(c) || is_number(c) || is_consonant(c) || *c == '_'))
        {
            if (is_vowel(c))
            {
                words_vowel_start_number++;
            }

            is_in_word = 1;
            words_number++;
        }

        if (is_separator(c) || is_vowel(c) || is_number(c) || is_consonant(c) || *c == '_' || *c == '\'')
        {
            memcpy(last_c, c, 4);
        }

        free(c);
    }

    free(last_c);
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