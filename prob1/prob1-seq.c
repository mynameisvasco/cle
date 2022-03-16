#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN_WORD_STATE 0
#define OUTSIDE_WORD_STATE 1

int are_utf8_chars_equal(char *c1, char *c2)
{
    int i = 0;
    while (i < 4)
    {
        if (*(c1 + 1) != *(c2 + 1))
            return 0;
        i++;
    }

    return 1;
}

int is_separator(char *c)
{
    char dash[4] = {0xE2, 0x80, 0x93, 0x00};
    return *c == '"' || *c == 0x20 || *c == 0x9 || *c == 0xA || *c == 0xD || *c == '.' || *c == '?' || *c == ';' || *c == ':' || *c == ',' || *c == '!' || are_utf8_chars_equal(c, dash);
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

int get_new_state(char *c, int current_state)
{

    if (current_state == OUTSIDE_WORD_STATE)
    {
        if (is_separator(c))
            return OUTSIDE_WORD_STATE;
        else
            return IN_WORD_STATE;
    }

    if (is_separator(c))
        return OUTSIDE_WORD_STATE;

    return IN_WORD_STATE;
}

void process_file(const char *file_path)
{
    int current_state = OUTSIDE_WORD_STATE;
    int words_number = 0;
    int words_vowel_start_number = 0;
    int words_consonant_ending_number = 0;
    FILE *file = fopen(file_path, "rb");

    if (file == NULL)
    {
        printf("File %s not found.\n", file_path);
        exit(-1);
    }

    while (1)
    {
        char *c = read_utf8_char(file);
        int new_state = get_new_state(c, current_state);

        if (new_state == IN_WORD_STATE && current_state == OUTSIDE_WORD_STATE)
            words_number++;

        if (*c == '\0')
            break;

        current_state = new_state;
        free(c);
    }

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

    for (int i = 1; i < argc; i++)
    {
        process_file(argv[i]);
    }
}