import judges


def test_version():
    assert judges.__version__ == '0.1.0'

if __name__ == '__main__':
    judges.fjc_create.main()
    judges.fjc_match.main()