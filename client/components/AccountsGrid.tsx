import AccountCard from './AccountCard';

interface AccountsGridProps {
  onAccountClick: (id: number) => void;
}

export default function AccountsGrid({ onAccountClick }: AccountsGridProps) {
  const accounts = [
    { id: 1, name: 'Account - 1' },
    { id: 2, name: 'Account - 2' },
    { id: 3, name: 'Account - 3' },
    { id: 4, name: 'Account - 4' }
  ];

  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="grid grid-cols-2 gap-8 max-w-2xl mx-auto">
          {accounts.map((account) => (
            <AccountCard
              key={account.id}
              id={account.id}
              name={account.name}
              onClick={onAccountClick}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
